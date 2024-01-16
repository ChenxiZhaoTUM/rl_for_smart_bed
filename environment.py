import gym
import numpy as np
from gym import spaces


class SmartBedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.alpha = 0.1
        self.episode = 1
        self.output_episode = 100
        self.time_per_action = 0.1

        self.action_space = spaces.MultiDiscrete([3] * 6)  # 0:no change 1:inflation 2:deflation
        low_obs = np.full(23, 0).astype(np.float32)
        high_obs = np.full(23, 1).astype(np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs)
        self.obs = np.zeros(23)

        self.ref_heart_rate = 60
        self.ref_breath_rate = 18

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_reward = 0.0
        self.action_time = 0.0
        self.action_time_steps = 0

        for i in range(16):
            self.obs[i] = 0  # pressure

        self.obs[16] = 0  # heart rate
        self.obs[17] = 0  # breath rate
        self.obs[18] = 0  # body motion
        self.obs[19] = 0  # snoring level
        self.obs[20] = 0  # in-bed power
        self.obs[21] = 0  # stability coefficient
        self.obs[22] = 0  # on bed or not

        self._get_obs = self.obs.astype(np.float32)
        print('train_obs: ', self._get_obs)

        return self._get_obs, {}

    def step(self, action):
        reward = 0.0
        done = False
        self.action_time_steps += 1
        self.action_time += self.time_per_action

        # action: 0:no change 1:inflation 2:deflation
        # take action of 6 airbags

        for i in range(16):
            self.obs[i] = 0  # pressure, need to be imported from the real-time measurement
        pressure_values = self.obs[:16]

        self.obs[16] = 0  # heart rate
        self.obs[17] = 0  # breath rate
        self.obs[18] = 0  # body motion
        self.obs[19] = 0  # snoring level
        self.obs[20] = 0  # in-bed power ??
        self.obs[21] = 0  # stability coefficient
        self.obs[22] = 1  # on bed or not

        if self.obs[22] == 0:
            done = True
        else:
            # set reward
            # pressure distribution
            pressure_variance = np.var(pressure_values)
            if pressure_variance == 0:
                reward += 10.0
            else:
                reward += 1.0 / pressure_variance

            # heart rate
            heart_rate_diff = np.abs(self.obs[16] - self.ref_heart_rate)
            if heart_rate_diff <= 5:
                reward += 1.0
            else:
                reward -= 1.0

            # breath rate
            breath_rate_diff = np.abs(self.obs[17] - self.ref_breath_rate)
            if breath_rate_diff <= 2:
                reward += 1.0
            else:
                reward -= 1.0

            # body motion
            reward += 1.0/self.obs[18]

            # snoring level
            reward += 1.0 / self.obs[19]

            # stability coefficient
            reward += 1.0 / self.obs[21]

        if done:
            self.episode += 1

        return self._get_obs, reward, done, False, {}


