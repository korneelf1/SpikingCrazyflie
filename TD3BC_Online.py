# TD3BC with jump-start (online-offline) for spiking actors.
import atexit
import multiprocessing as mp
import time
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from tqdm import tqdm

from spikingActorProb import SpikingNet
from tianshou.data import Batch, ReplayBuffer
from utils.directory_manager import save_checkpoint

ACTOR_OBS, PRIV_OBS, ACT = 18, 146, 4  # actor input, critic (privileged) obs, action sizes

# --- rollout workers (one env + one CPU actor per process; weights are shipped per task) ---
_W = {}


def _worker_init(env_kwargs, make_model, ctrl_state):
    from l2f_agent import ConvertedModel
    from l2f_gym import Learning2Fly
    torch.set_num_threads(1)
    ctrl = ConvertedModel()
    ctrl.load_state_dict(ctrl_state)
    _W.update(env=Learning2Fly(**env_kwargs), actor=make_model(), ctrl=ctrl)


@torch.no_grad()
def _rollout(env, actor, ctrl, js, explore, warmup, rollout_len, noise_std, noise_clip):
    """One episode. The first max(warmup, js) steps use the controller (the actor is only fed obs to warm up)."""
    obs = env.reset()[0]
    actor.reset_state()  # fresh membrane state per episode
    n_ctrl = max(warmup, js)
    obs_l, act_l, rew_l, done_l = [], [], [], []
    done = False
    while not done and len(obs_l) < rollout_len:
        o = torch.from_numpy(obs)
        if len(obs_l) < n_ctrl:
            action = ctrl(o)
            actor(o[:ACTOR_OBS])
        else:
            action = actor(o[:ACTOR_OBS]).reshape(ACT)
            if explore:
                action = (action + (torch.randn(ACT) * noise_std).clamp(-noise_clip, noise_clip)).clamp(-2, 2)
        action = action.numpy()
        obs_l.append(obs)
        obs, rew, done, _, _ = env.step(action)
        act_l.append(action.reshape(ACT))
        rew_l.append(rew)
        done_l.append(done)
    stack = np.hstack((np.array(obs_l), np.array(act_l), np.array(rew_l).reshape(-1, 1), np.array(done_l).reshape(-1, 1))).astype(np.float32)
    return stack, float(sum(rew_l))


def _collect(task):
    """Run n episodes with the given actor weights; in gather mode, retry episodes shorter than min_len."""
    state, env_params, n, mode, kw = task
    kw = dict(kw)
    min_len = kw.pop('min_len')
    env, actor = _W['env'], _W['actor']
    actor.load_state_dict(state)
    for k, v in env_params.items():
        setattr(env, k, v)
    out = []
    for _ in range(n):
        for _ in range(20):  # shortcut: gives up after 20 too-short episodes, the caller retries
            stack, total = _rollout(env, actor, _W['ctrl'], **kw)
            if mode == 'test' or len(stack) > min_len:
                out.append((stack if mode == 'gather' else None, total, len(stack)))
                break
    return out


class TD3BC_Online:
    def __init__(self, 
                 env, 
                 model,                 
                 optimizer,
                 critic1, 
                 critic1_optimizer, 
                 critic2, 
                 critic2_optimizer, 
                 buffer, 
                 make_model,
                 env_kwargs:dict|None = None,
                 n_workers:int = 0,
                 actor_obs_size:int = ACTOR_OBS, 
                 priveliged_obs_size:int = PRIV_OBS,
                 action_size:int = ACT,
                 batch_size:int = 256, 
                 warmup:int = 50, 
                 slicing_interval:int = 25,
                 sequence_length:int = 100,
                 device:str = 'cpu', 
                 controller:nn.Module|None = None, 
                 curriculum:bool = False,
                 bc_val:float = 1,
                 bc_factor:float = 0.95,
                 jumpstart_only_for_warmup:bool = False,
                 wandb_run=None, 
                 policy_noise:float = 0.2,
                 noise_clip:float = 0.5,
                 alpha:float = 2.5,
                 tau:float = 0.001,
                 freq:int = 2,
                 gamma:float = 0.99,
                 curriculum_update_interval:int = 30000
                 ):
        """
        make_model: picklable zero-arg factory building an actor with the same architecture as `model`
        (rollout workers load its weights each task). n_workers=0 collects in-process.
        """
        self.env = env
        self.device = device
        self.actor = model
        self.actor_obs_size = actor_obs_size
        self.priveliged_obs_size = priveliged_obs_size
        self.action_size = action_size
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.wandb_run = wandb_run
        self.slicing_interval = slicing_interval
        self.sequence_length = sequence_length
        if "terminated" in buffer._meta:
            # if terminated, truncated, dones and actions are 3 dims, squash to 2
            if len(buffer._meta.terminated.shape) == 3:
                buffer._meta.terminated = buffer.terminated[:,0,0]
            if len(buffer._meta.truncated.shape) == 3:
                buffer._meta.truncated = buffer.truncated[:,0,0]
            if len(buffer._meta.done.shape) == 3:
                buffer._meta.done = buffer.done[:,0,0]
            if len(buffer._meta.act.shape) == 3:
                buffer._meta.act = buffer.act[:,0]

        self.curriculum = curriculum  
        self.timestamp = time.strftime("%y%m%d-%H%M%S")
        print("original buffer size:", len(buffer))

        self.batch_size = batch_size
        self.warmup = warmup
        # if device is mps, no float64 can be used
        if device == 'mps':
            torch.set_default_dtype(torch.float32)

        self.alpha = alpha
        self._alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self._freq = freq
        self._cnt = 0
        self.curriculum_update_interval = curriculum_update_interval
        self.controller = controller
        # target networks
        self.critic1_old = deepcopy(critic1).to(device)
        self.critic2_old = deepcopy(critic2).to(device)
        self.actor_old = deepcopy(model).to(device)
        self.bc_coeff = bc_val
        self.bc_factor = bc_factor
        self.jumpstart_only_for_warmup = jumpstart_only_for_warmup

        self.last_test_rew = 0 # used for adaptive scheduling of slopes
        self.best_test = 0
        self.best_median_test_len = 0  # track best median test length for artifact logging
        self.envsteps = 0

        # TD3 adds noise to the target_q actions and evaluates using CURRENT policy! Seems to improve training
        self.policy_noise = policy_noise 
        self.noise_clip = noise_clip

        # rollout workers
        ctrl_state = {k: v.cpu() for k, v in controller.state_dict().items()}
        init = (env_kwargs or {}, make_model, ctrl_state)
        if n_workers > 0:
            self._pool = mp.get_context("spawn").Pool(n_workers, _worker_init, init)
            atexit.register(self._pool.terminate)
        else:
            self._pool = None
            _worker_init(*init)
        self.n_workers = n_workers

    def _run_tasks(self, n, mode, **kw):
        """Collect n episodes with the current actor, spread over the workers."""
        state = {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}
        env_params = {k: getattr(self.env, k) for k in ('Cp', 'Cv', 'Ca')}  # curriculum lives in the main env
        kw = dict(warmup=self.warmup, noise_std=self.policy_noise, noise_clip=self.noise_clip,
                  min_len=self.sequence_length, **kw)
        k = max(1, self.n_workers)
        chunks = [n // (4 * k)] * (4 * k) if n >= 4 * k else [1] * n
        chunks[0] += n - sum(chunks)
        tasks = [(state, env_params, c, mode, kw) for c in chunks if c > 0]
        res = self._pool.imap_unordered(_collect, tasks) if self._pool else map(_collect, tasks)
        return [r for chunk in res for r in chunk]

    def test(self, n_episodes:int = 30):
        res = self._run_tasks(n_episodes, 'test', js=0, explore=False, rollout_len=10**9)
        rewards = [r[1] for r in res]
        lengths = [r[2] for r in res]
        median_len = np.median(lengths)
        self.last_test_rew = avg_rew = np.mean(rewards)
        self.wandb_run.log({
            'test reward': avg_rew,
            'test len': np.mean(lengths),
            'test reward min': np.min(rewards),
            'test reward max': np.max(rewards),
            'test reward median': np.median(rewards),
            'test len min': np.min(lengths),
            'test len max': np.max(lengths),
            'test len median': median_len
        })
        if avg_rew > self.best_test:
            self.best_test = avg_rew
            filename = f"TD3BC_Online_TEMP_{self.wandb_run.name}_{self.timestamp}.pth"
            checkpoint_path = save_checkpoint(self.actor.state_dict(), filename)
            self.wandb_run.log_artifact(checkpoint_path, name='policy_streaming', type='model')
        
        # Log artifact based on median test length
        if median_len >= self.best_median_test_len:
            self.best_median_test_len = median_len
            filename = f"TD3BC_Online_MEDIAN_{self.wandb_run.name}_{self.timestamp}.pth"
            checkpoint_path = save_checkpoint(self.actor.state_dict(), filename)
            self.wandb_run.log_artifact(checkpoint_path, name='median_based_artifact', type='model')

    def _mse_optimizer(self, obs, act, target_q, critic, optimizer):
        """A simple wrapper script for updating critic network."""
        current_q = critic(obs.reshape(-1, self.priveliged_obs_size), act.reshape(-1, self.action_size)).flatten()
        critic_loss = (current_q - target_q.flatten()).pow(2).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        return critic_loss
    
    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters(), strict=True):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        self.soft_update(self.actor_old, self.actor, self.tau)

    def learn_batch(self, batch):
        P, A = self.priveliged_obs_size, self.action_size
        data = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)  # [B, T, obs|act|rew|done]
        B, T = data.shape[:2]
        observations, actions = data[..., :P], data[..., P:P+A]
        rewards, terminated = data[..., P+A], data[..., P+A+1]
        observations_next = torch.cat((observations[:, 1:], torch.zeros_like(observations[:, :1])), dim=1)
        
        # target actions with clipped noise (policy smoothing); the target actor runs over the sequence from a fresh state
        with torch.no_grad():
            self.actor_old.reset_state()
            pi_old = torch.stack([self.actor_old(observations[:, t, :self.actor_obs_size]) for t in range(T)], dim=1)
            target_actions_next = torch.cat((pi_old[:, 1:], pi_old[:, -1:]), dim=1)
            noise = torch.randn_like(target_actions_next) * self.policy_noise
            if self.noise_clip > 0.0:
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
            target_actions_next = target_actions_next + noise
            nxt, nxt_a = observations_next.reshape(-1, P), target_actions_next.reshape(-1, A)
            target_q = torch.min(self.critic1_old(nxt, nxt_a), self.critic2_old(nxt, nxt_a)).reshape(B, T)
        returns = rewards + self.gamma * (1 - terminated) * target_q
        
        # === CRITIC UPDATE ===
        critic_loss = self._mse_optimizer(observations, actions, returns, self.critic1, self.critic1_optimizer)
        critic2_loss = self._mse_optimizer(observations, actions, returns, self.critic2, self.critic2_optimizer)
        
        # === ACTOR UPDATE ===
        if self._cnt % self._freq == 0:
            self.optimizer.zero_grad()
            current_actions = torch.stack([self.actor(observations[:, t, :self.actor_obs_size]) for t in range(T)], dim=1)
            q_value = self.critic1(observations.reshape(-1, P), current_actions.reshape(-1, A)).reshape(B, T)
            q_value = q_value[:, self.warmup:]
            
            lmbda = self._alpha / q_value.abs().mean().detach()
            bc_loss = F.mse_loss(current_actions[:, self.warmup:], actions[:, self.warmup:])
            actor_loss = -(lmbda * q_value.mean() - self.bc_coeff * bc_loss)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()
            
            self.sync_weight()
            
            self.wandb_run.log({"actor_loss": actor_loss.item(),
                                "critic_loss": critic_loss.item(),
                                "critic2_loss": critic2_loss.item()})
        
        self._cnt += 1
    
    def learn(self, 
              epoch:int = 0, 
              end_epoch:int = 10):
        for n in tqdm(range(epoch, end_epoch)):
            self.wandb_run.log({"epoch":n})
            self.actor.to(self.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                # pass last test reward and current epoch to reset (used for adaptive scheduling and interval based scheduling of slopes, respectively)
                self.actor.reset(current_epoch = n,
                                 last_test_rew = self.last_test_rew)
                self.learn_batch(self.buffer.sample(self.batch_size)[0])
 
            if n%3==0:
                self.test()

    def gather_buffer(self, 
                      size:int = 2500, 
                      rollout_len:int = 501, 
                      jump_start_len:int|None = None):
        """Collect rollouts in parallel (episodes shorter than sequence_length are discarded), slice them into
        sequence_length windows (stride slicing_interval) and add them to self.buffer."""
        buffer_size = self.batch_size if self.batch_size > size + len(self.buffer) else size
        buffer = ReplayBuffer(size=buffer_size)
        per_rollout = (rollout_len - self.sequence_length) / self.slicing_interval + 1  # best case windows per episode
        seq = self.sequence_length
        stale = 0
        while len(buffer) < buffer.maxsize:
            n = int(np.ceil((buffer.maxsize - len(buffer)) / per_rollout))
            res = [r for r in self._run_tasks(n, 'gather', js=jump_start_len or 0, explore=True, rollout_len=rollout_len)]
            stale = 0 if res else stale + 1
            if stale >= 5:
                raise RuntimeError("actor crashes before sequence_length in every rollout; use a (longer) jump start")
            for stack, _, _ in res:
                self.envsteps += len(stack)
                windows = [stack[j:j+seq] for j in range(0, len(stack) - seq, self.slicing_interval)]
                if len(stack) % seq > 0:
                    windows.append(stack[-seq:])
                for w in windows:
                    P, A = self.priveliged_obs_size, self.action_size
                    buffer.add(Batch({'obs': w, 'act': w[-1, P:P+A], 'rew': float(w[-1, P+A]),
                                      'terminated': bool(w[-1, P+A+1]), 'truncated': bool(w[-1, P+A+1])}))
        self.wandb_run.log({'environment interactions': self.envsteps})
        self.buffer.update(buffer)
        return buffer
    
    def run(self, 
            jumpstart:bool = False,
            n_samples_per_gather:int = 200,
            max_epochs:int = 300,
            epochs_per_gather:int = 10
            ):
        """
        Alternate between gathering data (optionally jump-started by the controller) and learning.
        Logs to WandB and updates the reward curriculum if enabled.
        
        Args:
            jumpstart (bool): If True, the controller drives the first (500 - n) steps of each rollout, with n growing over training.
        """
        save_checkpoint(self.actor.state_dict(), f"TD3BC_Online_TEMP_{self.timestamp}_epoch_0.pth")
        rollout_len = 501
        factor_i = rollout_len/5/max_epochs # we want to be fully relying on the model by 80 percent of the end of the training
        n_samples_collected = 0
        curriculum_update_count = 0
        for i in range(0, max_epochs, epochs_per_gather):
            t0 = time.time()
            if jumpstart and not self.jumpstart_only_for_warmup:
                buffer = self.gather_buffer(jump_start_len=int(500-i*factor_i), size=n_samples_per_gather)
                self.wandb_run.log({"jump start steps (500 - n)": i})
            elif self.jumpstart_only_for_warmup:
                buffer = self.gather_buffer(jump_start_len=int(100), size=n_samples_per_gather)
                self.wandb_run.log({"jump start steps (100)": 100})
            else:
                buffer = self.gather_buffer(size=n_samples_per_gather)
            t1 = time.time()
            
            n_samples_collected += len(buffer)
            self.wandb_run.log({"n_samples_collected": n_samples_collected})
            self.wandb_run.log({"behavorial cloning coefficient": self.bc_coeff})
            self.learn(epoch=i, end_epoch=i+epochs_per_gather)
            print(f"[timing] cycle {i}: gather {t1-t0:.1f}s, learn+test {time.time()-t1:.1f}s", flush=True)
            self.bc_coeff *= self.bc_factor
            
            # Update curriculum every curriculum_update_interval collected samples
            if self.curriculum and curriculum_update_count - self.curriculum_update_interval > 0:
                self.env.update_curriculum()
                print(f"Curriculum updated at training cycle {curriculum_update_count}")
                curriculum_update_count = 0
            curriculum_update_count += n_samples_per_gather

class Wrapper(nn.Module):
    """Spiking body + tanh action head."""
    def __init__(self, model, size=128, action_size=4):
        super().__init__()
        self.preprocess = model
        self.mu = nn.Linear(size, action_size)
        # unused by forward(); kept so published checkpoints still load
        self.sigma = nn.Linear(size, action_size)
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.preprocess(x)
        if isinstance(x, tuple):
            x = x[0]    #spiking
        return torch.tanh(self.mu(x))
    def reset(self, **kwargs):
        if hasattr(self.preprocess, 'reset'):
            self.preprocess.reset(**kwargs)
    def reset_state(self):
        """Zero the membrane state only (reset() also steps the slope scheduler)."""
        self.preprocess.model.reset()


def make_actor(hidden_sizes, device="cpu", **snn_kwargs):
    """Spiking actor: SNN on the first hidden sizes, linear head from the last one."""
    snn = SpikingNet(state_shape=ACTOR_OBS, action_shape=hidden_sizes[-1], hidden_sizes=hidden_sizes[:-1],
                     device=device, reset_in_call=False, repeat=1, **snn_kwargs)
    return Wrapper(snn, size=hidden_sizes[-1]).to(device)


if __name__ == "__main__":
    import argparse
    import os

    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import Critic

    from l2f_agent import ConvertedModel
    from l2f_gym import Learning2Fly

    # set working directory as file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256,128])
        parser.add_argument("--actor-lr", type=float, default=5e-4)
        parser.add_argument("--critic-lr", type=float, default=3e-4)
        parser.add_argument("--n-step", type=int, default=1, help="unused, kept for old scripts")
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--buffer-size", type=int, default=20000, help="Buffer size")
        parser.add_argument("--buffer-preload", action='store_true', help="Buffer preload")

        parser.add_argument("--alpha", type=float, default=2.5)
        parser.add_argument("--policy-noise", type=float, default=0.1)
        parser.add_argument("--noise-clip", type=float, default=0.5)
        parser.add_argument("--update-actor-freq", type=int, default=2)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--gamma", type=float, default=0.99)

        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )
        parser.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="rollout processes, 0 = in-process")
        parser.add_argument("--wandb-project", type=str, default="l2f_bc")
        parser.add_argument("--curriculum", action='store_true', help="Enable reward curriculum scheduling")
        parser.add_argument("--jumpstart", action='store_true', help="JumpStartScheduling")
        parser.add_argument("--max_epochs", type=int, default=300, help="Max epochs")
        parser.add_argument("--epochs_per_gather", type=int, default=20, help="Epochs per gather")
        parser.add_argument("--slope", type=int, default=2, help="Slope value")
        parser.add_argument("--slope_schedule", type=str, default='adaptive')
        parser.add_argument("--scheduling_order", type=int, default=3)
        parser.add_argument("--bc-factor", type=float, default=0.99, help="Behavioral cloning factor")
        
        parser.add_argument("--bc-val", type=float, default=0.2, help="Behavioral cloning factor")
        parser.add_argument("--jumpstart_only_for_warmup", action='store_true', help="JumpStartScheduling only for warmup")
        parser.add_argument("--ablation", type=str, default=None)
        parser.add_argument("--n_rollouts_per_gather", type=int, default=10, help="Number of rollouts per gather")
        parser.add_argument("--stable_flight", action='store_true', help="Enable stable flight")
        parser.add_argument("--ang_vel_penalty", type=float, default=0.0, help="Angular velocity penalty")
        parser.add_argument("--survival_reward", type=float, default=1.0, help="Survival reward")
        return parser.parse_args()

    args = get_args()
    device = args.device

    controller = ConvertedModel()
    controller.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))

    env_kwargs = dict(fast_learning=False, stable_flight=args.stable_flight, ang_vel_penalty=args.ang_vel_penalty, survival_reward=args.survival_reward)
    env = Learning2Fly(**env_kwargs)

    buffer = ReplayBuffer(size=args.buffer_size)
    if args.buffer_preload:
        buffer_path = 'buffers/l2f_buffer_1996.hdf5'
        if not os.path.exists(buffer_path):
            raise FileNotFoundError(f"Buffer file not found: {buffer_path} (run install.sh to download it)")
        buffer.update(ReplayBuffer.load_hdf5(buffer_path))
    print(f"Loaded buffer with {len(buffer)} samples")

    wandb.init(project=args.wandb_project, config=dict(
        spiking=True, Algo='TD3BC_JS_Online', fast_learning=False, device=device, slope=args.slope,
        slope_schedule=args.slope_schedule, hidden_sizes=args.hidden_sizes, curriculum=args.curriculum,
        bc_factor=args.bc_factor, jumpstart=args.jumpstart, ablation=args.ablation,
        jumpstart_only_for_warmup=args.jumpstart_only_for_warmup))
    wandb.define_metric("*", step_metric="epoch")

    # Spiking actor; `snn_kw` is also what the rollout workers get (they never need the scheduler)
    model = make_actor(args.hidden_sizes, device=device, slope=args.slope, schedule=args.slope_schedule,
                       order=args.scheduling_order, reward_range=(-400,400), max_slope=20, verbose=True)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.actor_lr)

    def make_critic():
        net = Net(state_shape=PRIV_OBS, action_shape=ACT, hidden_sizes=args.hidden_sizes, concat=True, device=device)
        critic = Critic(net, device=device, flatten_input=False).to(device)
        return critic, torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    critic, critic_optim = make_critic()
    critic2, critic2_optim = make_critic()

    trainer = TD3BC_Online(env, model, optimizer,
                make_model=partial(make_actor, args.hidden_sizes, schedule='fixed'),
                env_kwargs=env_kwargs,
                n_workers=args.n_workers,
                controller=controller,
                critic1=critic,
                critic1_optimizer=critic_optim,
                critic2=critic2,
                critic2_optimizer=critic2_optim,
                buffer=buffer,
                batch_size=args.batch_size , device=device,
                curriculum=args.curriculum,
                bc_val=args.bc_val,
                bc_factor=args.bc_factor,
                jumpstart_only_for_warmup=args.jumpstart_only_for_warmup,
                wandb_run=wandb.run,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                alpha=args.alpha,
                tau=args.tau,
                freq=args.update_actor_freq,
                gamma=args.gamma,
                curriculum_update_interval=100000 if args.stable_flight else 30000)

    trainer.run(jumpstart=args.jumpstart, 
                n_samples_per_gather=args.n_rollouts_per_gather,
                max_epochs=args.max_epochs,
                epochs_per_gather=args.epochs_per_gather)
    
    wandb.run.finish()
    save_checkpoint(model.state_dict(), f'TD3BC_ONLINE_STABLE_{time.strftime("%y%m%d-%H%M%S")}.pth')
