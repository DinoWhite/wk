import numpy as np
import random
import matplotlib.pyplot as plt
import time

# 曼哈顿距离作为启发式函数 (用于 A* 算法)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 迷宫环境
class MazeEnv:
    def __init__(self, size=10, num_obstacles=20, try_limit=150):
        self.size = size
        self.num_obstacles = num_obstacles
        self.maze = np.zeros((size, size))
        self.state_size = size * size
        self.try_limit = try_limit
        self.reached_count = 0
        self.reset()

    def reset(self):
        self.maze = np.zeros((self.size, self.size))
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) != (0, 0) and (x, y) != (self.size - 1, self.size - 1):
                self.obstacles.add((x, y))

        for obstacle in self.obstacles:
            self.maze[obstacle] = -1  # -1 表示障碍物

        while True:
            self.agent_pos = [
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            ]
            self.goal_pos = [
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            ]
            if (
                tuple(self.agent_pos) not in self.obstacles
                and tuple(self.goal_pos) not in self.obstacles
                and self.agent_pos != self.goal_pos
            ):
                break
        self.maze[self.goal_pos[0], self.goal_pos[1]] = 2  # 2 代表目标
        self.steps = 0
        return self.get_state()

    def step(self, action):
        self.steps += 1
        new_pos = list(self.agent_pos)

        if action == 0:  # 向上
            new_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # 向下
            new_pos[0] = min(self.agent_pos[0] + 1, self.size - 1)
        elif action == 2:  # 向左
            new_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # 向右
            new_pos[1] = min(self.agent_pos[1] + 1, self.size - 1)

        if tuple(new_pos) not in self.obstacles:
            self.agent_pos = new_pos

        reward = -0.1
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 5
            done = True
        elif self.steps >= self.try_limit:
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        return state.flatten()

    def render(self):
        maze_copy = np.copy(self.maze)
        maze_copy[self.agent_pos[0], self.agent_pos[1]] = 1
        plt.imshow(maze_copy, cmap="hot", interpolation="nearest")
        plt.pause(0.005)
        plt.clf()

# Q-learning 代理
class QLearningAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.q_table = np.zeros((state_size, action_size))  # 初始化Q-table
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.1

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # ε-greedy 策略：随机选择动作（探索）
            return random.randrange(self.action_size)
        # 利用：选择 Q-table 中 Q 值最大的动作
        state_idx = self.get_state_idx(state)
        return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state, done):
        state_idx = self.get_state_idx(state)
        next_state_idx = self.get_state_idx(next_state)

        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        self.q_table[state_idx, action] += self.learning_rate * (target - self.q_table[state_idx, action])

        # 逐渐减少 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_state_idx(self, state):
        # 将迷宫状态展平成一维索引
        state = state.reshape(self.env.size, self.env.size)
        return np.ravel_multi_index(np.unravel_index(state.argmax(), (self.env.size, self.env.size)), (self.env.size, self.env.size))

# 主程序
if __name__ == "__main__":
    grid_size = 10
    num_obstacles = 20
    try_limit = 150
    env = MazeEnv(size=grid_size, num_obstacles=num_obstacles, try_limit=try_limit)
    state_size = env.state_size
    action_size = 4  # 上、下、左、右
    agent = QLearningAgent(state_size, action_size, env)
    episodes = 1000
    episode_rewards = []
    avg_rewards = []

    plt.ion()

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        for time in range(try_limit):
            env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
                break

        episode_rewards.append(total_reward)

        # 每100次统计一次平均reward
        if (e + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e+1}: Avg Reward in last 100 episodes: {avg_reward}, Epsilon: {agent.epsilon}")

    plt.ioff()

    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward (Last 100 Episodes)")
    plt.show()
