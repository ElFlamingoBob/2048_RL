import gymnasium as gym
from model import Model
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import time

env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

model = Model() 

episodes = 5000

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
		return 1
	else:
		return -1

start_time = time.time()

for episode in range(episodes):
	observation, info = env.reset()
	illegal_move_count = 0
	last_move = None
	same_move_count = 0
	rewards.clear()

	while True:
		state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
		action = model.select_action(state, training=True)
		observation, reward, terminated, truncated, info = env.step(action.item())

		reward = float(reward)
		if reward > 0.0:
			reward = math.log2(reward)

		if last_move == action.item():
			same_move_count += 1
		else:
			same_move_count = 0

		if same_move_count >= 10:
			reward = -1.0

		last_move = action.item()
		actions[int(action.item())] += 1
		steps += 1
		if info.get("is_legal") is False:
			illegal_move_count += 1
			reward = -10.0
			terminated = True
		else:
			additive = 0.0
			for i in range(observation.shape[1]):
				for j in range(observation.shape[0]):
					additive += 0.2 if observation[i][j][0] == 1 else 0
			additive = round(additive, 1)
			reward  = float(reward) + additive if float(reward) > 0 else float(reward)
			reward += max_in_corners(info["board"])

		rewards.append(reward)


		# print(reward)

		next_state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0) if not terminated else None
		ts_reward = torch.tensor([reward], dtype=torch.float32, device=model.device)

		model.memory.push(state, action, next_state, ts_reward)
		state = next_state
		loss = model.optimize_model()

		model.soft_update_target_net()

		if terminated or truncated:
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


plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("plots/training_loss.png")
plt.close()

plt.plot(rewards_means)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Reward per Episode")
plt.savefig("plots/average_reward.png")
plt.close()

plt.plot(illegal_moves)
plt.xlabel("Episode")
plt.ylabel("Illegal Moves")
plt.title("Illegal Moves per Episode")
plt.savefig("plots/illegal_moves.png")
plt.close()

plt.plot(actions_in_time)
plt.xlabel("Episode (every 100 episodes)")
plt.ylabel("Action Distribution")
plt.title("Action Distribution Over Time")
plt.legend(["Up", "Down", "Left", "Right"])
plt.savefig("plots/action_distribution.png")
plt.close()

model.save_model("model.pth")
end_time = time.time()

print(f"Training completed in {round(end_time - start_time, 2)} seconds. Model saved as 'model.pth'. (steps taken: {steps})")
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