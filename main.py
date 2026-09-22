import os
import time
os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
from model import Model
import torch


env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

model = Model() 

model.load_model("model_5k.pth")

observation, info = env.reset()
terminated, truncated = False, False

grid = info["board"]
for row in grid:
	print([(int(2 ** (value - 1 if value > 0 else 0)) if value > 0 else 0) for value in row])
print(" ")

actions = ["up", "right", "down", "left"]

while terminated is False and truncated is False:
	state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
	action = model.select_action(state, training=False)
	observation, reward, terminated, truncated, info = env.step(action.item())

	print(f"Action: {actions[int(action.item())]}, Reward: {reward}, Legal Move: {info.get('is_legal')}")

	grid = info["board"]
	for row in grid:
		print([(int(2 ** (value - 1 if value > 0 else 0)) if value > 0 else 0) for value in row])
	print(terminated, truncated)
	print(" ")

	# input("Press Enter to continue...")
	



env.close() 