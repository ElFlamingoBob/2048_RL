import os
import time
os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
from model import Model
import torch


env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

model = Model() 

model.load_model("model.pth")

observation, info = env.reset()
terminated, truncated = False, False

while terminated is False and truncated is False:
	state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
	action = model.select_action(state, training=False)
	observation, reward, terminated, truncated, info = env.step(action.item())

	grid = info["board"]
	for row in grid:
		print([int(2 ** value) for value in row])
	print(" ")



env.close() 