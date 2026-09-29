import os
import time
os.environ["SDL_AUDIODRIVER"] = "dummy"

import gymnasium as gym
from model import Model
from resnet_model import ResNetModel
import torch



def main():

	# env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16, render_mode="human")  # Create the 2048 environment with specified parameters
	env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

	# model = Model()
	model = ResNetModel()
	# model.load_model("__model_7k_end_ft.pth")
	# model.load_model("__model_10k_end_ft.pth")
	# model.load_model("cdqn_10k.pth")
	model.load_model("resnet_10k.pth")
	max_score = 0

	score_repartion = [0] * 17
	number_of_games = 100

	for _ in range(number_of_games):

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
		# print(2 ** (score))

		if score > max_score:
			max_score = score

		score_repartion[score] += 1

		# input("Press Enter to continue...")

	env.close() 

	print(f"Max Score: {2 ** (max_score)}")
	for i in range(len(score_repartion)):
		if score_repartion[i] > 0:
			print(f"Score {2 ** (i)}: {score_repartion[i] / number_of_games * 100:.2f}%")

main()