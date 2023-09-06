import numpy as np
import gym
import matplotlib.pyplot as plt

m, l, g = 1., 1., 9.8
torq_max = 1.5
target_cos = -1.
target_time = 10.

low_action = [-1.]
high_action = [1.]
low_state = [0., -np.pi, -6.]
high_state = [target_time, np.pi, 6.]

class Env(gym.Env):
    def __init__(self, dt=0.01):
        super(Env, self).__init__()

        # Setting for system
        self.dt = dt
        self.idx = 0
    
        self.time = np.arange(0., target_time + self.dt, self.dt)
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
        self.reset()

    def reset(self):
        self.episodes += 1
        self.idx = 0
        
        self.time = np.arange(0., target_time + self.dt, self.dt)
        self.theta = np.zeros_like(self.time)
        self.omega = np.zeros_like(self.time)
        self.torque = np.zeros_like(self.time)

        #print(self.episodes, self.theta[self.idx], self.omega[self.idx], self.torque[self.idx])
        return np.array([self.time[self.idx], self.theta[self.idx], self.omega[self.idx]])

    def step(self, action):
        self.idx += 1

        self.torque[self.idx] = action[0] * torq_max
        omega_t = (self.torque[self.idx] - m * g * l * np.sin(self.theta[self.idx - 1])) / (m * l * l)
        self.omega[self.idx] = self.omega[self.idx - 1] + self.dt * omega_t
        self.theta[self.idx] = self.theta[self.idx - 1] + self.dt * self.omega[self.idx] + 0.5 * self.dt ** 2 * omega_t

        print(self.time[self.idx], self.theta[self.idx], self.omega[self.idx], self.torque[self.idx])
        if self.time[self.idx] >= target_time:
            reward = - (np.cos(self.theta[self.idx]) - target_cos) ** 2
            done = True
            fig, ax = plt.subplots(1, 1, figsize=(5, 1.5))
            ax.plot(self.time, self.theta, 'k', label='Angle')
            ax.plot(self.time, self.torque, 'b', label='Torque')
            
            ax.axhline(np.pi, c='r', ls='--', label='Goal')
            ax.axhline(-np.pi, c='r', ls='--')
            ax.axhline(0, c='k', ls='--', lw=0.5, zorder=-1)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('rl_evolution.svg')
            plt.show()

            
        else:
            reward = 0.
            done = False
        
        return np.array([self.time[self.idx], self.theta[self.idx], self.omega[self.idx]]), reward, done, {}

    def render(self, mode = 'human'):
        pass

    def close(self):
        pass

