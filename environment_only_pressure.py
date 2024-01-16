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

        self.previous_pressure_values = np.zeros(16)
        self.previous_action = np.zeros(6)  # here need to change to the previous inner pressure of the airbag

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

        pressure_variance = np.var(pressure_values)

        pressure_change_continuity = np.mean(np.abs(self.previous_pressure_values - pressure_values))
        self.previous_pressure_values = pressure_values.copy()

        action_change_continuity = np.mean(np.abs(self.previous_action - action))
        self.previous_action = action.copy()

        # set reward
        # pressure distribution
        if pressure_variance == 0:
            reward += 10.0
        else:
            reward += 1.0 / pressure_variance

        reward -= pressure_change_continuity
        reward -= action_change_continuity

        if done:
            self.episode += 1

        return self._get_obs, reward, done, False, {}
