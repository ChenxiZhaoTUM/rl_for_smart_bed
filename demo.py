import numpy as np


class SmartBedEnv:
    def __init__(self):
        # Assuming self.obs is a NumPy array containing 16 dimensions
        self.obs = np.zeros(16)

    def calculate_normalized_reward(self):
        # Extract the first 8 dimensions representing pressure values
        pressure_values = self.obs[:8]

        # Calculate the variance of the pressure values
        pressure_variance = np.var(pressure_values)

        # Log transformation to map variance to [0, 1]
        # normalized_reward = np.log(1 + pressure_variance)

        # Normalize to [0, 1]
        # normalized_reward /= np.log(1 + np.var(np.ones(8)))

        # Clip the normalized reward to ensure it stays within [0, 1]
        # normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

        return 1/pressure_variance


# Example usage:
env = SmartBedEnv()

# Set some example pressure values in the observation space
env.obs[:8] = np.array([1.0, 2.0, 1.5, 1.2, 1.8, 1.3, 2.5, 200.0])

# Calculate and print the normalized reward
normalized_reward = env.calculate_normalized_reward()
print("Normalized Reward:", normalized_reward)
