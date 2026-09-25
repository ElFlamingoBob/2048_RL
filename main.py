import os
import time
os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
from model import Model
import torch



def main():

	env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

	model = Model()
	model.load_model("_model_5k_end.pth")
	max_score = 0

	for _ in range(100):

		observation, info = env.reset()
		terminated, truncated = False, False

		# grid = info["board"]
		# for row in grid:
		# 	print([(int(2 ** (value - 1 if value > 0 else 0)) if value > 0 else 0) for value in row])
		# print(" ")

		actions = ["up", "right", "down", "left"]

		def is_illegal_move(board):
			actions = [0, 0, 0, 0]  # up, right, down, left
			x, y = board.shape
			for i in range(x):
				for j in range(y):
					if i > 0 and ((board[i][j] == board[i-1][j] and board[i][j] != 0) or (board[i][j] != 0 and board[i-1][j] == 0)):
						actions[0] = 1
					if j < 3 and ((board[i][j] == board[i][j+1] and board[i][j] != 0) or board[i][j] != 0 and board[i][j+1] == 0):
						actions[1] = 1
					if i < 3 and ((board[i][j] == board[i+1][j] and board[i][j] != 0) or (board[i][j] != 0 and board[i+1][j] == 0)):
						actions[2] = 1
					if j > 0 and ((board[i][j] == board[i][j-1] and board[i][j] != 0) or board[i][j] != 0 and board[i][j-1] == 0):
						actions[3] = 1
			return actions

		while terminated is False and truncated is False:
			state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
			action = model.select_action(state, is_illegal_move(info["board"]), training=False)
			observation, reward, terminated, truncated, info = env.step(action.item())

			# print(f"Action: {actions[int(action.item())]}, Reward: {reward}, Legal Move: {info.get('is_legal')}")

		grid = info["board"]
		# for row in grid:
		# 	# print([(int(2 ** (value - 1 if value > 0 else 0)) if value > 0 else 0) for value in row])
		# 	print([int(value) for value in row])
		# print(" ")
		
		
		score = max([int(value) for row in grid for value in row])
		print(2 ** (score - 1))

		if score > max_score:
			max_score = score

		# input("Press Enter to continue...")

	env.close() 

	print(f"Max Score: {2 ** (max_score - 1)}")

main()