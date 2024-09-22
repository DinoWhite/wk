import gym
import numpy as np
import matplotlib.pyplot as plt
import random


class GameEnv(gym.Env):
    def __init__(self):
        self.size = 41  # 地图大小41x41
        self.map = np.zeros((self.size, self.size))  # 初始化地图，0表示空地
        self.water_areas = []  # 用来存储水区位置

    def reset(self):
        self.map = np.zeros((self.size, self.size))  # 重置地图为空地
        self._generate_water_areas()  # 生成水域
        return self.map

    def _generate_water_areas(self):
        # 生成2到3个水区
        num_water_areas = random.randint(2, 3)
        for _ in range(num_water_areas):
            # 随机生成一个水区的大小，范围在40到100个方格之间
            water_size = random.randint(30, 80)
            # 随机选择一个起点位置（确保不会在边界上）
            start_x = random.randint(1, self.size - 2)
            start_y = random.randint(1, self.size - 2)

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
                if 0 <= new_x < self.size and 0 <= new_y < self.size:
                    water_area.add((new_x, new_y))

            # 把生成的水域标记在地图上，1表示水域
            for x, y in water_area:
                self.map[x, y] = 1  # 1表示水
            self.water_areas.append(water_area)

    def render(self):
        plt.imshow(self.map, cmap="Blues")  # 用蓝色显示水域
        plt.title("Game Environment with Water Areas")
        plt.show()


# 调用示例
env = GameEnv()
env.reset()  # 生成一个新的游戏环境
env.render()  # 渲染当前环境

# 让环境显示，直到用户按 ESC 或关闭窗口
plt.show()
