from gymnasium import Env

class SpikingEnv(Env):
    """ A wrapper that just modifies the reset function
    """
    def __init__(self, env, spiking_model=None):
        self.env = env
        self.spiking_model = spiking_model

    def reset(self):
        if self.spiking_model is not None:
            self.spiking_model.reset()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.seed(seed)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def __str__(self):
        return str(self.env)

    def __repr__(self):
        return repr(self.env)

    def __len__(self):
        return len(self.env)

    def __getitem__(self, key):
        return self.env[key]

    def __setitem__(self, key, value):
        self.env[key] = value

    def __delitem__(self, key):
        del self.env[key]

    def __iter__(self):
        return iter(self.env)

    def __contains__(self, item):
        return item in self.env

    def __eq__(self, other):
        return self.env == other

    def __ne__(self, other):
        return self.env != other

    def __bool__(self):
        return bool(self.env)

    def __hash__(self):
        return hash(self.env)

    def __enter__(self):
        return self.env.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.env.__exit__(exc_type, exc_value, traceback)

    def __call__(self, *args, **kwargs):
        return self.env(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def __str__(self):
        return str(self.env)

    def __repr__(self):
        return repr(self.env)

    def __len__(self):
        return len(self.env)

    def __getitem__(self, key):
        return self.env[key]

    def __setitem__(self, key, value):
        self.env[key] = value

    def __delitem__(self, key):
        del self.env[key]

    def __iter__(self):
        return iter(self.env)

    def __contains__(self, item):
        return item in self.env

    def __eq__(self, other):
        return self.env == other

    def __ne__(self, other):
        return self.env != other
    
    