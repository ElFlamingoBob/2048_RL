import gymnasium as gym
from model import Model
from resnet_model import ResNetModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import time

env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

# model = Model()
model = ResNetModel()

# model.load_model("10k_base.pth")
start_episode = 0
# start_episode = model.load_checkpoint("models/resnet/checkpoint.pth")  # Load the model from the specified path

episodes = 10000

loss_not_improving_count = 0
last_loss = float('inf')
losses = []
illegal_moves = []
steps = 0

rewards = []
rewards_means = []

actions = [0, 0, 0, 0]
actions_in_time = []

def max_in_corners(board):
	board = np.array(board)
	max_value = board.max()
	if board[0][0] == max_value or \
		  board[0][-1] == max_value or \
			board[-1][0] == max_value or \
				board[-1][-1] == max_value:
		return 3
	else:
		return -3

def test_model(model, num_games=100):

	eval_env = gym.make(
		"gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0",
		size=4,
		max_pow=16,
	)

	model.policy_net.eval()
	score_repartition = [0] * 17

	for _ in range(num_games):
		observation, info = eval_env.reset()
		terminated, truncated = False, False

		while terminated is False and truncated is False:
			state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
			action = model.select_action(state, is_illegal_move(info["board"]), training=False)
			observation, reward, terminated, truncated, info = eval_env.step(action.item())

		score = max([int(value) for row in info["board"] for value in row])
		score_repartition[score] += 1

	eval_env.close()
	model.policy_net.train()

	for i, count in enumerate(score_repartition):
		if count > 0:
			print(f"Score {2 ** i}: {count / num_games * 100:.2f}%")

# 3 3 1 2 x
# 2 1 0 0
# 0 4 2 1
# 1 0 0 0
# y

# x = 0
# y = 0

# def is_illegal_move(board):
# 	board = np.array(board)

# 	def can_move(current, neighbor):
# 		return np.any(
# 			(current != 0)) & \
# 			((current == neighbor) | ((neighbor == 0)) 
# 		)

# 	return [
# 		int(can_move(board[1:, :], board[:-1, :])),  # up
# 		int(can_move(board[:, :-1], board[:, 1:])),  # right
# 		int(can_move(board[:-1, :], board[1:, :])),  # down
# 		int(can_move(board[:, 1:], board[:, :-1])),  # left
# 	]

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

start_time = time.time()

for episode in range(start_episode, episodes):
	observation, info = env.reset()
	illegal_move_count = 0
	last_move = None
	same_move_count = 0
	rewards.clear()

	legal_moves = is_illegal_move(info["board"])


	while True:
		state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
		action = model.select_action(state, legal_moves, training=True)
		action = torch.tensor([[action]], device=model.device, dtype=torch.long)
		observation, reward, terminated, truncated, info = env.step(action.item())
		legal_moves = is_illegal_move(info["board"])

		reward = float(reward)
		if reward > 0.0:
			reward = math.log2(reward) * 1.5

		# if last_move == action.item():
		# 	same_move_count += 1
		# else:
		# 	same_move_count = 0

		# if same_move_count >= 10:
		# 	reward = -1.0

		last_move = action.item()
		actions[int(action.item())] += 1
		steps += 1
	
		additive = 0.0
		for i in range(observation.shape[1]):
			for j in range(observation.shape[0]):
				additive += 0.7 if observation[i][j][0] == 1 else 0
		additive = round(additive, 2)
		reward = float(reward) + additive if float(reward) > 0 else float(reward)
		reward += max_in_corners(info["board"]) if float(reward) > 0 else 0

		rewards.append(reward)

		game_finished = terminated or truncated
		# print(reward)

		next_state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0) if not game_finished else None
		next_mask = torch.tensor([legal_moves], dtype=torch.bool, device=model.device) if not game_finished else None
		ts_reward = torch.tensor([reward], dtype=torch.float32, device=model.device)

		model.memory.push(state, action, next_state, ts_reward, next_mask)
		state = next_state
		loss = model.optimize_model()

		model.soft_update_target_net()

		if game_finished:
			rewards_means.append(sum(rewards) / len(rewards))
			illegal_moves.append(illegal_move_count)
			losses.append(loss)
			break

	# if loss != None and loss >= last_loss:
	# 	loss_not_improving_count += 1
	# else:
	# 	loss_not_improving_count = 0

	last_loss = loss if loss != None else last_loss

	# if loss_not_improving_count >= 15:
	# 	print("Loss has not improved for 15 consecutive steps. Stopping training.")
	# 	break
	# print(f"Episode: {episode}, Loss: {round(last_loss, 2)}, loss_nic: {loss_not_improving_count}")

	if episode % 100 == 0:
		actions_in_time.append(actions.copy())
		print(f"Episode {episode} completed. (steps: {steps})")

	if episode % 300 == 0 and episode > 0:
		torch.save(model.policy_net.state_dict(), f"models/resnet/resnet_{episode / 300}.pth")
		model.save_checkpoint("models/resnet/checkpoint.pth", episode + 1)
		test_model(model, num_games=100)

model.save_checkpoint(
	"models/resnet/checkpoint_final.pth",
	episodes + 1,
)

plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("plots/training_loss_resnet.png")
plt.close()

plt.plot(rewards_means)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Reward per Episode")
plt.savefig("plots/average_reward_resnet.png")
plt.close()

plt.plot(illegal_moves)
plt.xlabel("Episode")
plt.ylabel("Illegal Moves")
plt.title("Illegal Moves per Episode")
plt.savefig("plots/illegal_moves_resnet.png")
plt.close()

plt.plot(actions_in_time)
plt.xlabel("Episode (every 100 episodes)")
plt.ylabel("Action Distribution")
plt.title("Action Distribution Over Time")
plt.legend(["Up", "Down", "Left", "Right"])
plt.savefig("plots/action_distribution_resnet.png")
plt.close()

model.save_model("models/resnet/resnet_10k.pth")
end_time = time.time()

print(f"Training completed in {round(end_time - start_time, 2)} seconds. Model saved as 'resnet_10k.pth'. (steps taken: {steps})")
print(f"Action distribution: Up: {actions[0]}, Down: {actions[1]}, Left: {actions[2]}, Right: {actions[3]}")


env.close() 

# observation, info = env.reset()
# print(observation.shape)
# print(observation[0][0])
# for i in range(25):
# 	random_action = env.action_space.sample()
# 	observation, reward, terminated, truncated, info = env.step(random_action)
# 	if info.get("is_legal") is False:
# 		reward = -10.0
# 	else:
# 		multiplier = 0.0
# 		for i in range(observation.shape[1]):
# 			for j in range(observation.shape[0]):
# 				print(observation[i][j])
# 				multiplier += 0.1 if observation[i][j][0] == 1 else 0
# 		multiplier = round(multiplier, 1) + 1 
# 		reward = float(reward) * multiplier
# 		print(f"Reward: {reward}, Multiplier: {multiplier}")
# 	if info.get("is_legal") is False:
# 		print("Illegal move attempted. Assigning penalty.")

# env.close()