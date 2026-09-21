import math

import torch
import torch.nn as nn
import torch.optim as optim

import random

from collections import namedtuple, deque

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class CDQN(nn.Module):
	def __init__(self):
		super(CDQN, self).__init__()
		self.layer1 = nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=0)
		self.layer2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)
		self.layer3 = nn.Flatten()
		self.layer4 = nn.Linear(512, 256)
		self.layer5 = nn.Linear(256, 4)

	def forward(self, x):
		x = torch.relu(self.layer1(x))
		x = torch.relu(self.layer2(x))
		x = self.layer3(x)
		x = torch.relu(self.layer4(x))
		return self.layer5(x)


class Model:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.memory = ReplayMemory(300000)
		self.batch_size = 64
		self.gamma = 0.99
		self.epsilon_start = 0.9
		self.epsilon_end = 0.01
		self.epsilon_decay = 65000
		self.policy_net = CDQN().to(self.device)
		self.target_net = CDQN().to(self.device)
		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.001)

		self.steps_done = 0

	def select_action(self, state, training=False):

		if training:
			sample = random.random()
			epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
				(math.exp(-1. * self.steps_done / self.epsilon_decay))
			self.steps_done += 1

			if sample > epsilon_threshold:
				with torch.no_grad():
					return self.policy_net(state).max(1)[1].view(1, 1)
			else:
				return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
		else:
			with torch.no_grad():
				return self.policy_net(state).max(1)[1].view(1, 1)

	def optimize_model(self):
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

		state_action_values = \
			self.policy_net(state_batch).gather(1, action_batch.view(-1, 1))

		next_state_values = torch.zeros(self.batch_size, device=self.device)

		with torch.no_grad():
			next_state_values[non_final_mask] = \
				self.target_net(non_final_mask_states).max(1).values

		expected_state_action_values = \
			(next_state_values * self.gamma) + reward_batch

		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values,
						 expected_state_action_values.unsqueeze(1))

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()

	def soft_update_target_net(self, tau=0.005):
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
		self.epsilon_start = 0.1
		self.steps_done = 0
	

