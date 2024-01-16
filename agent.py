import random
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            # placeholder
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity  # 使用模运算（%）来循环覆盖缓冲区的旧样本，确保缓冲区不会无限增长，而是保持固定容量 self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # input: batch_size * 4D (state)
    # output: batch_size * 2D (action)
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)


class Agent:
    def __init__(self, n_states, n_actions, eta=0.5, gamma=0.99, capacity=10000, batch_size=32):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = ReplayMemory(capacity)
        self.model = DQN(n_states, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)  # 从经验回放缓冲区（Experience Replay Buffer）中随机抽样一批经验样本以供训练
        batch = Transition(*zip(*batch))  # 将list of transition 转化成一个transition, column: len(tuple) == batch_size

        state_batch = torch.cat(batch.state)  # s_t为4D，即s_t.shape == batch_size * 4 (pos, cart_v, angle, pole_v)
        action_batch = torch.cat(batch.action)  # a_t.shape == batch_size * 1
        reward_batch = torch.cat(batch.reward)  # r_{t+1}.shape == batch_size * 1
        # next_state_batch = torch.cat(batch.next_state)
        non_final_next_state_batch = torch.cat(
            [s for s in batch.next_state if s is not None])  # 将符合条件的s拼接成一个张量，可能小于batch_size，即有些遇到结束状态

        # 构造model的input和output
        # input: s_t
        # pred: Q(s_t, a_t)
        # true: R_{t+1} + gamma * max(Q(s_{t+1}, a))
        # purpose: pred approach true

        # 开启eval模式
        self.model.eval()

        # pred
        # 维度: batch_size * 1
        state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)

        # true
        # tuple(map(lambda s: s is not None, batch.next_state)) 为batch_size长度的0/1
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))  # 用于标识批量中哪些样本具有非空的下一个状态next_state (1)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_state_batch).max(dim=1)[
            0].detach()  # 计算非终止状态下的下一状态值，max(dim=1)[0]为状态值的最大估计

        # 维度: (batch_size, )
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # 开启train模式
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))  # 增加一个维度对齐

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def update_q_function(self):
        self._replay()

    def choose_action(self, state, episode):
        eps = 0.5 * 1 / (1 + episode)

        if random.random() < eps:
            # explore
            action = torch.IntTensor([[random.randrange(self.n_actions)]])  # 将action转化成2D，与action_space维度相同
        else:
            # exploit
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(dim=1)[1].view(1, 1)
                # self.model(state)计算状态的Q值估计，max(dim=1)选择具有最高Q值估计的动作的索引，view(1, 1)将索引变成一个2D张量以便与选择的动作格式一致
        return action
