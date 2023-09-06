import numpy as np
import gym

m, l, g = 1., 1., 9.8
torq_max = 1.5
target_cos = -1.
target_time = 10.
dt = 0.01
skip = 5

low_action = [-1.]
high_action = [1.]
low_state = [0., -np.pi, -6.]
high_state = [target_time, np.pi, 6.]

class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()

        # Setting for system
        self.time = np.arange(0., target_time + dt, dt)
        self.theta = np.zeros_like(self.time)
        self.omega = np.zeros_like(self.time)
        self.torque = np.zeros_like(self.time)

        # Setting for RL
        self.action_space = gym.spaces.Box(
            low = np.array(low_action),
            high = np.array(high_action),
            dtype = np.float32
        )
        self.observation_space = gym.spaces.Box(
            low = np.array(low_state),
            high = np.array(high_state),
            dtype = np.float32
        )

        # Initialize
        self.episodes = 0
        self.idx = 0

    def reset(self):
        self.episodes += 1
        self.idx = 0
        
        self.theta = np.zeros_like(self.time) + np.random.uniform(-np.pi, np.pi)
        self.omega = np.zeros_like(self.time) + np.random.uniform(-1., 1.)
        self.torque = np.zeros_like(self.time)
        return np.array([self.time[self.idx], self.theta[self.idx], self.omega[self.idx]])

    def step(self, action):
        for _ in range(skip):
            self.idx += 1
            self.torque[self.idx] = action[0] * torq_max
            omega_t = (self.torque[self.idx] - m * g * l * np.sin(self.theta[self.idx - 1])) / (m * l * l)
            self.omega[self.idx] = self.omega[self.idx - 1] + dt * omega_t
            self.theta[self.idx] = self.theta[self.idx - 1] + dt * self.omega[self.idx] + 0.5 * dt ** 2 * omega_t

        if self.time[self.idx] >= target_time:
            reward = - (np.cos(self.theta[self.idx]) - target_cos) ** 2
            done = True
            print(self.episodes, self.theta[self.idx], self.omega[self.idx], self.torque[self.idx], reward)
        else:
            reward = 0.
            done = False
        
        return np.array([self.time[self.idx], self.theta[self.idx], self.omega[self.idx]]), reward, done, {}

    def render(self, mode = 'human'):
        pass

    def close(self):
        pass

