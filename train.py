import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class SoldierEnv(gym.Env):
    def __init__(self):
        super(SoldierEnv, self).__init__()

        # 地图大小为41x41
        self.grid_size = 41
        self.max_steps = 1000  # 每个回合的最大步数

        # 状态空间包括：小兵的坐标、资源点、敌人的位置等
        # 假设状态为小兵的(x, y)坐标和资源的(x, y)坐标
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(4,), dtype=np.int32
        )

        # 动作空间：8个方向的移动
        # 0: 上, 1: 下, 2: 左, 3: 右, 4: 左上, 5: 右上, 6: 左下, 7: 右下
        self.action_space = spaces.Discrete(8)

        # 初始化小兵和资源的位置
        self.reset()

        # 初始化可视化窗口
        self.fig, self.ax = plt.subplots()
        self.ax.grid(True, color="black", linestyle="-", linewidth="0.5")
        self.ax.set_xticks(np.arange(0, self.grid_size + 1, 1))  # 刻度从 0 到 grid_size，步长为 1
        self.ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.soldier_patch = Rectangle((0, 0), 1, 1, color="blue")  # 小兵位置的方块
        self.resource_patch = Rectangle((0, 0), 1, 1, color="green")  # 资源位置的方块
        self.ax.add_patch(self.soldier_patch)
        self.ax.add_patch(self.resource_patch)

    def reset(self):
        # 重置状态
        self.soldier_position = np.random.randint(0, self.grid_size, size=(2,))
        self.resource_position = np.random.randint(0, self.grid_size, size=(2,))
        self.steps = 0
        return np.concatenate([self.soldier_position, self.resource_position])

    def step(self, action):
        # 根据动作更新小兵位置
        self.steps += 1
        move = self.get_move(action)
        self.soldier_position = np.clip(
            self.soldier_position + move, 0, self.grid_size - 1
        )

        # 计算奖励
        done = False
        reward = -0.1  # 每步有小的负奖励以鼓励小兵快速找到资源
        if np.array_equal(self.soldier_position, self.resource_position):
            reward = 10  # 找到资源
            done = True  # 任务完成

        if self.steps >= self.max_steps:
            done = True  # 达到最大步数

        return (
            np.concatenate([self.soldier_position, self.resource_position]),
            reward,
            done,
            {},
        )

    def get_move(self, action):
        # 根据动作返回移动的方向
        if action == 0:  # 上
            return np.array([0, 1])
        elif action == 1:  # 下
            return np.array([0, -1])
        elif action == 2:  # 左
            return np.array([-1, 0])
        elif action == 3:  # 右
            return np.array([1, 0])
        elif action == 4:  # 左上
            return np.array([-1, 1])
        elif action == 5:  # 右上
            return np.array([1, 1])
        elif action == 6:  # 左下
            return np.array([-1, -1])
        elif action == 7:  # 右下
            return np.array([1, -1])

    def render(self, mode="human"):
        # 可视化地图（简单的打印位置）
        print(f"Soldier: {self.soldier_position}, Resource: {self.resource_position}")
        # 更新小兵和资源的位置
        self.soldier_patch.set_xy(self.soldier_position)
        self.resource_patch.set_xy(self.resource_position)

        self.ax.set_title(f"Step: {self.steps}")
        plt.pause(0.01)  # 暂停一段时间，以便更新图像


import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 超参数
GAMMA = 0.99  # 折扣因子
LR = 1e-3  # 学习率
BATCH_SIZE = 64  # 训练批量大小
MEMORY_SIZE = 10000  # 经验池大小
TARGET_UPDATE = 10  # 每隔多少步更新目标网络
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.01  # 最小探索率
EPSILON_DECAY = 0.995  # 探索率衰减


# DQN代理类
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_q_network(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q值更新公式
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# 训练过程
def train_dqn(env, agent, episodes):
    rewards = []  # 用于记录每个回合的总奖励

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(1000):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_q_network()

            state = next_state
            total_reward += reward

            env.render()  # 渲染当前状态（可视化）

            if done:
                break

        agent.update_target_network() if episode % TARGET_UPDATE == 0 else None

        # Epsilon 衰减
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        print(f"Episode {episode}, Total Reward: {total_reward}")
    # 绘制奖励变化曲线
    plt.figure()
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


# 初始化环境和代理
env = SoldierEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 训练1000个回合
train_dqn(env, agent, episodes=1000)
