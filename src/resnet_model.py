import math

import torch
import torch.nn as nn
import torch.optim as optim

import random

from collections import namedtuple, deque

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward', 'next_mask'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class ResidualBlock(nn.Module):
	def __init__(self):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)

	def forward(self, x):
		resisual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += resisual
		out = self.relu(out)

		return out

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet, self).__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(16, 128, kernel_size=3, padding=1, stride = 1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
		)

		blocks = []
		for _ in range(4):
			blocks.append(ResidualBlock())
		self.residual_blocks = nn.Sequential(*blocks)

		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(128 * 4 * 4, 256)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(256, 4)

	def forward(self, x):
		x = self.stem(x)
		x = self.residual_blocks(x)
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x
		


class ResNetModel:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.memory = ReplayMemory(300000)
		self.batch_size = 128
		self.gamma = 0.99
		self.epsilon_start = 0.9
		self.epsilon_end = 0.01
		self.epsilon_decay = 500000
		self.policy_net = ResNet().to(self.device)
		self.target_net = ResNet().to(self.device)
		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.0001, weight_decay=0.0001)

		self.steps_done = 0

	def select_action(self, state, is_illegal_move, training=False):
		self.policy_net.eval()
		if training:
			sample = random.random()
			epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
				(math.exp(-1. * self.steps_done / self.epsilon_decay))
			self.steps_done += 1

			if sample > epsilon_threshold:
				with torch.no_grad():
					output = self.policy_net(state).detach().cpu().numpy()[0]
					for i in range(4):
						output[i] = -math.inf if is_illegal_move[i] == 0 else output[i]
					return output.argmax()
			else:
				return torch.tensor([[random.choice([i for i in range(4) if is_illegal_move[i] == 1])]], device=self.device, dtype=torch.long)
		else:
			with torch.no_grad():
				# output = self.policy_net(state).max(1)[1].view(1, 1)
				output = self.policy_net(state).detach().cpu().numpy()[0]
				for i in range(4):
					output[i] = -math.inf if is_illegal_move[i] == 0 else output[i]
				# print(f"Output: {output}, Legal Moves: {is_illegal_move}")
				return output.argmax()

	def optimize_model(self):
		self.policy_net.train()
		if len(self.memory) < self.batch_size:
			return
		transitions = self.memory.sample(self.batch_size)

		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(
			tuple(map(lambda s: s is not None, batch.next_state)),
			dtype=torch.bool,
			device=self.device)
		non_final_mask_states = torch.cat(
			[s for s in batch.next_state if s is not None]
		)

		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		next_mask_batch = torch.cat([m for m in batch.next_mask if m is not None])

		state_action_values = \
			self.policy_net(state_batch).gather(1, action_batch.view(-1, 1))

		next_state_values = torch.zeros(self.batch_size, device=self.device)

		with torch.no_grad():
			target_q_values = self.target_net(non_final_mask_states)
			target_q_values[~next_mask_batch] = -math.inf
			next_state_values[non_final_mask] = target_q_values.max(1).values

		expected_state_action_values = (next_state_values * self.gamma) + reward_batch

		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values,
						 expected_state_action_values.unsqueeze(1))

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()

		return loss.item()

	def soft_update_target_net(self, tau=0.001):
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = tau * policy_net_state_dict[key] + (1 - tau) * target_net_state_dict[key]
		self.target_net.load_state_dict(target_net_state_dict)

	def save_model(self, path):
		torch.save(self.policy_net.state_dict(), path)

	def load_model(self, path):
		self.policy_net.load_state_dict(
			torch.load(path, map_location=self.device, weights_only=True))
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.epsilon_start = 0.4
		self.steps_done = 0

	def get_epsilon(self):
		return self.epsilon_end + (
				self.epsilon_start - self.epsilon_end
		) * math.exp(-self.steps_done / self.epsilon_decay)

	def save_checkpoint(self, path, episode):
		torch.save({
			'episode': episode,
			'policy_net': self.policy_net.state_dict(),
			'target_net': self.target_net.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'steps_done': self.steps_done,
			'epsilon': self.get_epsilon()
		}, path)

	def load_checkpoint(self, path):

		checkpoint = torch.load(path, map_location=self.device, weights_only=False)
		self.policy_net.load_state_dict(checkpoint['policy_net'])
		self.target_net.load_state_dict(checkpoint['target_net'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.steps_done = checkpoint['steps_done']

		print(f"Loaded episode: {checkpoint['episode']}")
		print(f"Loaded epsilon: {self.get_epsilon():.6f}")
		return checkpoint['episode']
	

