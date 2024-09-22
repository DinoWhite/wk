import numpy as np
import torch
'''
我有一个问题，请帮我参谋一下怎么解决，我先告知你一下大致游戏规则：
首先我介绍一些基本规则和信息：
1. 环境内各元素间距离和攻击范围、控制范围等均按照切比雪夫距离计算
2. 游戏环境为41*41的矩阵，以第14、28列、14、28行将地图划分成9块区域，以从左到右，从上到下分辨命名为：LT区，T区、RT区，L区、C区、R区、LB区、B区、RB区，分界线不属于这9个区域，资源物品也不会在分界线中出现
3. 环境内部含有障碍物、敌方角色、我方角色、资源物品、boss角色等元素，环境内某个格子只允许放置一种元素，不可以叠放，除了资源可以和我方角色或者敌方角色可以叠放，因为角色要收集资源必须要与资源的坐标点重合
4. 游戏分为A型任务和B型任务，每种任务都有三个主要任务和两个日常任务，日常任务可以反复完成以兑换金币和血量，血量有上限，金币无上限
5. 我方角色和敌方角色类型相同，只是阵营不同，分为小兵，长官，元帅三种，小兵只能进行普通攻击（范围3，伤害3），长官可以普通攻击（范围4，伤害5）和技能攻击（范围3，伤害7），元帅可以普通攻击（范围5，伤害8）、技能攻击（范围5，伤害9）和技能控制（范围6）
6. 普通攻击只能对单独指定的目标造成伤害，技能攻击对攻击范围内全体非我方角色均可造成伤害，但是若其中被障碍物（山，寺庙）阻拦则对目标不构成伤害，阻拦判定方式为bresenham线，水不会阻挡攻击；技能控制不受障碍物阻挡影响，在范围内既对非我方角色生效，技能控制效果为非我方角色3回合内无法做任何操作
7. 山和水所在坐标任何角色均无法站立和经过
8. 对手和boss会移动，移动速度为1格/时间片，移动时若被障碍物阻挡则无法移动，水会阻挡角色移动，山、寺庙、防御塔、基地会阻挡角色攻击和移动
9. 资源物品有3种，木头、石头、鱼肉、鸡肉，收集一块木头+一个石头交付到寺庙进行兑换操作即可获得5个金币，收集一个鱼肉+一个鸡肉到寺庙兑换可以恢复60血量，对方角色和我方角色均会收集资源和兑换，资源不会在障碍物上刷新
10. 一个回合有两个时间片可以执行一个操作，一个操作消耗一个时间片，例如一个回合内，一个角色第一个时间片移动一步走到了资源上方，然后第二个时间片进行收集
11. boss是环境指定位置的，boss不会移动，boss会对自己攻击范围内的角色进行攻击，若有多个角色（敌方和我方都可能进入范围）在boss攻击范围内，先进入的角色会被boss锁定，boss只会对被锁定的角色攻击，直到被锁定的角色离开boss的攻击范围后boss才会换角色锁定攻击
12. 每个玩家角色（敌方角色，我方角色）都有各自的背包，背包内装单个角色独自收集的物品，金币是全队共享的，例如我方小兵A背包找到木头后，走到寺庙兑换后获得金币5个，则我方金币+5，小兵A背包里木头减一，兑换血量也是只补充给兑换的人，每个角色背包最多放5个资源
13. 双方角色的元帅不可以互相攻击，其他角色均可攻击，即敌方元帅不可以攻击我方元帅，我方也不可攻击对方元帅，但是我方小兵和长官可以攻击对方元帅
14. 两个队伍初始都有三个小兵血量150，背包为空，每收集并兑换一个木头或鱼肉，积分+10，每购买一个角色积分+30，每摧毁对方角色一个积分+100，每摧毁对方防御塔一个积分+150，摧毁基地积分+300，每摧毁一个boss积分+500，游戏总回合数500
15. A型B型任务都有三个阶段任务，两队500回合内先把任务都做完的为赢家，500回合结束均未做完则按照积分高低评判胜负
16. 寺庙是用于兑换金币和血量的地点，普通攻击、技能攻击和角色均不能穿过寺庙，寺庙也属于障碍物，角色要在寺庙周围才可进行兑换
17. 队伍可以使用金币购买小兵，长官和元帅，小兵10金币，长官20金币，元帅30金币，需要购买时只需向环境请求即可，环境确认队伍金币足够后会在地图中随机位置生成该队伍的元帅或元帅，同时扣除该队伍对应的金币数目
18. 长官和元帅也可以收集资源进行兑换
19. B型任务中还有基地和防御塔建筑，只有长官可以制造基地和防御塔，基地占地3*3的格子，基地所处范围均算障碍物，基地和防御塔会对敌方角色发动攻击，攻击逻辑和boss一样，会先锁定进入的第一个敌方角色，防御塔占地一个方格，也是障碍物
20. 基地血量1500，攻击范围以基地为中心距离5，防御塔血量800，攻击范围为2
21. 角色单次移动可向8项移动，斜向移动的情况下，若目标位置没有障碍物即可移动过去，比如向右上斜向移动，此时右侧有障碍物，上方也有障碍物，但是右上方没有，则此时是可以移动成功的
22. 若向有障碍物的方位移动，角色本次移动后仍停留在原地
23. 若角色有交换位置形式的移动，则也无法移动成功，例如角色A位于（3，3），角色B位于（3，4），此时A向上移动，目标位置(3,4),B向下移动，目标位置(3,3)则此次移动结算后，A任然在（3，3），B也在（3，4）
24. 若角色有目标移动位置相同的场景，也无法移动成功，例如角色A位于（3，3），角色B位于（3，5），此时A向上移动,目标位置（3，4），B向下移动，目标位置(3,4),则此次移动结算后，A任然在（3，3），B也在（3，5）
25. 游戏开始时会生成地图和20个左右的资源（每种资源为4到6个）、两个寺庙，以及各队三个小兵，小兵初始位置随机，资源位置也随机
26. 资源被某一队取走则在地图上标记消失，每50个回合地图自动清除场上未被收集的资源，并重新随机生成20个左右的资源点（每种资源为4到6个）
27. 地图向右为x轴正方向，向上为y轴正方向
28. 角色这些操作均消耗一个时间片（移动，攻击，技能控制，建造防御塔，建造基地，技能攻击，收集，兑换，丢弃）
29. 角色击败boss会让元帅回满血量，并且元帅的攻击伤害会增加2
30. 小兵血量150，长官血量250，元帅血量400，角色拥有数量有限制(已有数+死亡角色数)，小兵最多买5个，长官最多1个，元帅最多一个
31. B型任务中防御塔和基地是由长官检查队伍在金币数足够且周围有足够空间建造的情况下向系统购买后当场建造，购买不消耗时间片，建造消耗
32. 建造基地时，可建造范围为本队长官所处坐标点距离为2的方框上，因为基地占地为3*3，所以要向外延伸一格范围，同时要确保建造点为中心的3*3格子内没有角色、资源和障碍物
33. 建造防御塔时，可建造范围为本队长官所处坐标点的环绕一周的8个点，建造时要确保建造点没有角色、资源和障碍物等才可以建造
34. 长官和元帅发动技能控制需要额外消耗金币，金币不足则不能发动，技能攻击消耗3个金币，发动技能攻击后，不管是在哪个时间片发动的，都要在3回合后才允许再次发动技能攻击，技能控制需要消耗5个金币，不管是在哪个时间片发动的，都要在8回合后才允许再次发动技能攻击
35. 若建造时判定金币不足或空间不够，则本时间片内不能做其他操作，即认定做了一次建造动作，不过结果是失败的
36. 发动技能时，若金币不够，则不扣除金币，本时间片内不能做其他操作，即认定做了一次发动技能，不过结果是失败的
37. 日常任务可以反复完成来获得多次金币和血量，在每个阶段都可以做日常任务
38. 死亡角色会在地图上移除
39. 角色可消耗一个时间片丢弃包内一样物品，被丢弃的物品会被系统随机在地图上摆放

接下来介绍A型B型任务各阶段任务：
A型任务（收集资源并打boss）
1. 阶段1：初始双队均有三个满血小兵，初始每队20金币，第一阶段要尽快收集够30个金币，然后收集到30个金币后系统自动会消耗30金币购买元帅，购买元帅后，进入阶段2
2. 阶段2：系统会为本队生成一只本队负责击败的boss，叫流氓，血量1200，攻击范围3，攻击伤害3，本阶段任务就是要尽快击败流氓，击败流氓后进入阶段3
3. 阶段3：系统会在阶段3生成一只本队负责击败的boss，叫坏蛋一号，血量1000，攻击范围3，攻击伤害4，击败坏蛋一号后，系统又会生成坏蛋二号，血量1200，攻击范围4，攻击伤害4，击败坏蛋二号后，系统生成坏蛋三号，攻击范围4，攻击伤害5，击败坏蛋三号后，游戏结束

A型任务注意点：
1. 本队负责击败的boss也可被对面队伍击败，对面击败本队负责击败的boss的话，本队进入下一阶段，相当于帮本队忙了
2. 场上会有两个同名boss，但是一个是自己负责击败的，一个是对手负责的，注意不要帮对面打了

B型任务（建造己方基地和防御塔，摧毁对方基地和防御塔）
1. 阶段1：初始双队均有三个满血小兵，和一个满血长官，初始每队20金币，第一阶段要尽快让长官建造基地，长官建造好基地后进入下一个阶段2
2. 阶段2：尽快摧毁对手的基地和防御塔，摧毁对手基地则游戏结束

环境每回合会将游戏内的山水资源、角色、建筑信息都发送给每个队伍，请问针对上述规则，你推荐我只用什么算法和方式让自己的队伍取得高分赢得游戏

针对如上游戏规则，可以先帮我用matplotlib生成一个游戏环境吗，封装成gym的格式，我调用reset时会给我生成一个环境，render时会渲染出来环境,
生成水的时候让水方格都聚在一起，占地40到100个方格左右，全局只要生成2到3个水区就行，位置离交界线的4个交界点近一点，水区用浅蓝色填充；生成山的时候贴着水区环绕生成，将水区包围半圈左右，包围圈的方格厚度2到4格左右，再在其他敌方生成一些聚集在一起，不规则的山区，每个山区有1到50个不等的格子，生成8到9个山区即可，山区用浅灰色格子填充
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# 游戏环境大小 41x41
grid_size = 41

# 初始化两个角色的位置
pos1 = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]
pos2 = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]

# 定义角色的颜色和标记
char1_color = 'red'
char2_color = 'blue'

# 随机移动函数，每次移动一个方格
def random_move(pos):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右移动
    move = random.choice(moves)
    new_x = np.clip(pos[0] + move[0], 0, grid_size - 1)  # 防止越界
    new_y = np.clip(pos[1] + move[1], 0, grid_size - 1)
    return [new_x, new_y]

# 初始化画布和网格
fig, ax = plt.subplots()
ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.set_xlim(-0.5, grid_size-0.5)
ax.set_ylim(-0.5, grid_size-0.5)
ax.set_aspect('equal')

# 画两个角色
char1 = ax.plot(pos1[0], pos1[1], 's', color=char1_color, markersize=10)[0]
char2 = ax.plot(pos2[0], pos2[1], 's', color=char2_color, markersize=10)[0]

# 更新函数，每帧移动角色
def update(frame):
    global pos1, pos2
    pos1 = random_move(pos1)  # 随机移动角色1
    pos2 = random_move(pos2)  # 随机移动角色2
    char1.set_data(pos1[0], pos1[1])  # 更新角色1位置
    char2.set_data(pos2[0], pos2[1])  # 更新角色2位置
    return char1, char2

# 动画函数，每帧间隔为500毫秒
ani = animation.FuncAnimation(fig, update, frames=200, interval=500, blit=True)

plt.show()
