import gymnasium as gym
from model import Model
import torch

env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)  # Create the 2048 environment with specified parameters

model = Model() 

episodes = 1000

for episode in range(episodes):
	observation, info = env.reset()

	while True:
		state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0)
		action = model.select_action(state, training=True)
		observation, reward, terminated, truncated, info = env.step(action.item())

		if info.get("is_legal") is False:
			reward = -100.0
		else:
			additive = 0.0
			for i in range(observation.shape[1]):
				for j in range(observation.shape[0]):
					additive += 0.1 if observation[i][j][0] == 1 else 0
			additive = round(additive, 1)
			reward  = float(reward) + additive

		# print(reward)
		if terminated or truncated:
			break

		next_state = torch.tensor(observation, dtype=torch.float32, device=model.device).permute(2, 0, 1).unsqueeze(0) if not terminated else None
		ts_reward = torch.tensor([reward], dtype=torch.float32, device=model.device)

		model.memory.push(state, action, next_state, ts_reward)
		state = next_state
		model.optimize_model()
		model.soft_update_target_net()

	if episode % 10 == 0:
		print(f"Episode {episode} completed.")

model.save_model("model.pth")

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