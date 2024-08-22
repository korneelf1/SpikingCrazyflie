from l2f import Rng, Device, Environment, Parameters, State, Observation, Action,initialize_environment,step,initialize_rng,parameters_to_json,sample_initial_parameters,initial_state
import gymnasium as gym
import numpy as np

class Learning2Fly(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        # L2F initialization
        self.device = Device()
        self.rng = Rng()
        self.env = Environment()
        self.params = Parameters()
        self.state = State()
        self.next_state = State()
        self.observation = Observation()
        self.next_observation = Observation()
        self.action = Action()
        initialize_environment(self.device, self.env, self.params)
        initialize_rng(self.device, self.rng, 0)

        # Gym initialization
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,))
        self.global_step_counter = 0   
        self.t = 0

        self.reset()

    def step(self, action):
        self.action.motor_command = action
        step(self.device, self.env, self.params, self.state, self.action, self.next_state, self.rng)
        self.state = self.next_state

        self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)
        self.t += 1

        done = self._check_done()
        reward = self._reward()

        return self.obs, reward, done,done, {}
    
    def reset(self,seed=None):
        sample_initial_parameters(self.device, self.env, self.params, self.rng)
        initial_state(self.device, self.env, self.params, self.state)
        self.global_step_counter += self.t
        self.t = 0
        return np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32), {}
    
    def _reward(self):
        # intial parameters
        Cp = 1 # position weight
        Cv = .0 # velocity weight
        Cq = 0 # orientation weight
        Ca = .0 # action weight og .334, but just learns to fly out of frame
        Cw = .0 # angular velocity weight 
        Crs = .2 # reward for survival
        Cab = 0.0 # action baseline

        # curriculum parameters
        Nc = 1e5 # interval of application of curriculum

        CpC = 1.2 # position factor
        Cplim = 20 # position limit

        CvC = 1.4 # velocity factor
        Cvlim = 1.5 # velocity limit

        CaC = 1.4 # orientation factor
        Calim = .5 # orientation limit

        pos   = self.obs[0:3]
        vel   = self.obs[3:6]
        q     = self.obs[6:10]
        qd    = self.obs[10:13]

        # curriculum
        if self.global_step_counter % Nc == 0:
            Cp = min(Cp*CpC, Cplim)
            Cv = min(Cv*CvC, Cvlim)
            Ca = min(Ca*CaC, Calim)

        # r[0] = max(-1e3,-Cp*np.sum((pos-pset)**2) \
        #         - Cv*np.sum((vel)**2) \
        #             - Ca*np.sum((motor_commands-Cab)**2) \
        #                 - Cw*np.sum((qd)**2) \
        #                     + Crs)
        # print("pos penalty: ", -Cp*np.sum((pos)**2))
        # print("vel penalty: ", - Cv*np.sum((vel)**2))
        # print("action penalty: ", - Ca*np.sum((motor_commands-Cab)**2))
        # print("orientation penalty: ", -Cq*(1-q[0]**2)) aims to keep upright!
        # print("angular velocity penalty: ", - Cw*np.sum((qd)**2))
        # print("reward for survival: ", Crs)

        # in theory pos error max sqrt( .6)*2.5 = 1.94
        # vel error max sqrt(1000)*.005 = 0.158
        # qd error max sqrt(1000)*.00 = 0.
        # should roughly be between -2 and 2
        r = - Cv*np.sum((vel)**2) \
                - Ca*np.sum((np.array(self.action.motor_command)-Cab)**2) \
                    -Cq*(1-q[0]**2)\
                    - Cw*np.sum((qd)**2) \
                        + Crs \
                            -Cp*np.sum((pos)**2) 
        return r
    
    def _check_done(self):
        done = False
        pos_threshold = np.sum((np.abs(self.obs[0:3])>1.5))
        velocity_threshold = np.sum((np.abs(self.obs[3:6]) > 1000))
        angular_threshold  = np.sum((np.abs(self.obs[10:13]) > 1000))
        time_threshold = self.t>500

        # print("pos_threshold: ", pos_threshold)
        # print("velocity_threshold: ", velocity_threshold)
        # print("angular_threshold: ", angular_threshold)
        # print("time_threshold: ", time_threshold)

        if any(np.isnan(self.obs)):
            done = True
        elif pos_threshold or velocity_threshold or angular_threshold or time_threshold:
            done = True
            

        return done


if __name__=='__main__':
    from stable_baselines3.common.env_checker import check_env
    env = Learning2Fly()
    check_env(env)