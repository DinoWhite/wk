import numpy as np
import random
import matplotlib.pyplot as plt
import time


class MazeEnv:
    def __init__(self, size=10, num_obstacles=10):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        # 生成随机起点和终点
        self.start = (
            random.randint(0, self.size - 1),
            random.randint(0, self.size - 1),
        )
        self.end = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))

        # 生成随机障碍物
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            obstacle = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )
            if obstacle != self.start and obstacle != self.end:
                self.obstacles.add(obstacle)

        self.state = self.start
        return self.state

    def step(self, action):
        # 计算下一个状态
        next_state = list(self.state)
        if action == 0:  # 向上
            next_state[1] = max(0, self.state[1] - 1)
        elif action == 1:  # 向下
            next_state[1] = min(self.size - 1, self.state[1] + 1)
        elif action == 2:  # 向左
            next_state[0] = max(0, self.state[0] - 1)
        elif action == 3:  # 向右
            next_state[0] = min(self.size - 1, self.state[0] + 1)
        next_state = tuple(next_state)

        # 检查障碍物
        if next_state in self.obstacles:
            next_state = self.state

        # 检查奖励
        reward = -1
        done = False
        if next_state == self.end:
            reward = 100
            done = True

        self.state = next_state
        return next_state, reward, done

    def render(self):
        maze = np.zeros((self.size, self.size))
        maze[self.start] = 2  # 起点
        maze[self.end] = 3  # 终点
        for ob in self.obstacles:
            maze[ob] = 1  # 障碍物
        maze[self.state] = 4  # 当前状态
        plt.imshow(maze)
        plt.pause(0.005)  # 显示0.2秒
        plt.clf()  # 清除当前图像以准备显示下一步


# Q-learning agent
class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, state_size, action_size))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # 探索
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # 利用

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = (
            reward
            + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
        )
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.lr * td_error

        # 逐渐减少 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 训练过程
env = MazeEnv(size=10, num_obstacles=10)
agent = QLearningAgent(state_size=10, action_size=4)

episodes = 1000
max_steps = 100
rewards = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0

    for time in range(max_steps):
        env.render()  # 可视化每一步
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)
    if (e + 1) % 100 == 0:
        print(f"Episode: {e + 1}, Average Reward: {np.mean(rewards[-100:])}")

# 绘制训练奖励的折线图
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.show()
