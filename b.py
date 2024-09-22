import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random


class GameEnv(gym.Env):
    def __init__(self):
        super(GameEnv, self).__init__()

        # 游戏环境大小
        self.grid_size = 41

        # 初始化地图（0表示空地，1表示障碍物）
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # 障碍物，水，山等元素
        self.obstacles = []
        self.water = []
        self.mountains = []

        # 寺庙
        self.temples = []

        # 角色位置 (随机生成)
        self.friendly_units = []
        self.enemy_units = []

        # 资源
        self.resources = []

        # 游戏渲染使用的画布和轴
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")

    def reset(self):
        """
        重置环境，生成新的地图、资源、角色等元素
        """
        # 清空当前环境状态
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # 随机生成障碍物、山、水等
        # self.obstacles = self.generate_random_elements(10, "obstacle")
        self.generate_water_areas()
        self.mountains = self.generate_random_elements(200, "mountain")

        # 寺庙生成（两个寺庙随机生成）
        self.temples = self.generate_random_elements(2, "temple")

        # 随机生成友方和敌方角色
        self.friendly_units = self.generate_random_units(3, "friendly")
        self.enemy_units = self.generate_random_units(3, "enemy")

        # 随机生成资源
        self.resources = self.generate_random_elements(20, "resource")

        return self.grid

    def generate_random_elements(self, count, element_type):
        """
        随机生成元素，避免生成在同一位置或重叠
        """
        elements = []
        for _ in range(count):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if self.grid[y, x] == 0:  # 空地
                    self.grid[y, x] = 1  # 占据此位置
                    elements.append((x, y))
                    break
        return elements

    def generate_water_areas(self):
        # 生成2到3个水区
        num_water_areas = random.randint(2, 3)
        for _ in range(num_water_areas):
            # 随机生成一个水区的大小，范围在40到100个方格之间
            water_size = random.randint(30, 80)
            # 随机选择一个起点位置（确保不会在边界上）
            start_x = random.randint(1, self.grid_size - 2)
            start_y = random.randint(1, self.grid_size - 2)

            # 用队列算法生成聚集的水域
            water_area = set()
            water_area.add((start_x, start_y))
            while len(water_area) < water_size:
                # 从已有的水格子中随机扩展新的水格子
                x, y = random.choice(list(water_area))
                # 随机扩展的方向，上下左右
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_x, new_y = x + direction[0], y + direction[1]

                # 确保新坐标在地图范围内，且尚未是水
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    water_area.add((new_x, new_y))

            # 把生成的水域标记在地图上，1表示水域
            for x, y in water_area:
                self.grid[x, y] = 1  # 1表示水
            self.water.append(water_area)

    def generate_random_units(self, count, unit_type):
        """
        随机生成角色
        """
        units = []
        for _ in range(count):
            while True:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                if self.grid[y, x] == 0:  # 空地
                    self.grid[y, x] = 1  # 占据此位置
                    units.append((x, y))
                    break
        return units

    def render(self, mode="human"):
        """
        渲染当前环境，包括障碍物、资源、角色等
        """
        self.ax.clear()

        # 重新设置网格和边界
        self.ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_aspect("equal")

        # 渲染障碍物
        for x, y in self.obstacles:
            self.ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))

        # 渲染水
        for x, y in self.water:
            self.ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="blue"))

        # 渲染山
        for x, y in self.mountains:
            self.ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="gray"))

        # 渲染寺庙
        for x, y in self.temples:
            self.ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="yellow"))

        # 渲染资源
        for x, y in self.resources:
            self.ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="green"))

        # 渲染友方角色
        for x, y in self.friendly_units:
            self.ax.plot(x, y, "go", markersize=10, label="Friendly Unit")

        # 渲染敌方角色
        for x, y in self.enemy_units:
            self.ax.plot(x, y, "ro", markersize=10, label="Enemy Unit")

        # 刷新图像
        plt.draw()
        plt.pause(0.1)


env = GameEnv()
env.reset()  # 生成一个新的游戏环境
env.render()  # 渲染当前环境

plt.show()
