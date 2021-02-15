import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import NullFormatter

def minimal_by(eles, f):
    '''
    求eles中使f(x)最小元素min_x
    '''
    value = np.inf
    min_x = None
    for x in eles:
        tmp = f(x)
        if tmp < value:
            value = tmp
            min_x = x
    return min_x


def find_path(M, start, end):
    '''
    使用A*算法计算地图M中从start到end的最短路径，返回路径列表
    M为矩阵，M[i, j] > 0代表(i, j)地块不可通行，小于等于0代表可通行
    start, end均为坐标
    '''
    if M[start[0], start[1]] > 0 or M[end[0], end[1]] > 0:
        # 起点或终点不可通行，返回空列表
        return []
    open_dict = {start: 0}
    close_dict = {}
    parent = {}
    parent[start] = (start[0], start[1] - 1)
    while len(open_dict) > 0:
        x = minimal_by(
            open_dict, lambda x: open_dict[x] + abs(end[0] - x[0]) + abs(end[1] - x[1]))
        if x == end:
            break
        else:
            close_dict[x] = open_dict.pop(x)
            # 优先向右移动；优先向地图中线移动
            x_prefer = 1 if x[0] < 3 else -1
            for direct in [[0, 1], [x_prefer, 0], [-x_prefer, 0], [0, -1]]:
                y = (x[0] + direct[0], x[1] + direct[1])
                # 是否在地图范围内
                if not(0 <= y[0] < M.shape[0] and 0 <= y[1] < M.shape[1]):
                    continue
                # 是否可通行
                elif M[y[0], y[1]] > 0:
                    continue
                elif y not in close_dict and ((y not in open_dict) or (open_dict.get(parent[y], 0) > close_dict[x] + 1)):
                    parent[y] = x
                    open_dict[y] = close_dict[x] + 1
    if end in parent:
        path = [end]
        x = parent[end]
        while x in parent:
            path.append(x)
            x = parent[x]
        return path[::-1]
    else:
        return []


def calculate_map(M0, max_box=4):
    # 所有可放置箱子地块
    zero_pos = np.c_[(M0 == 0).nonzero()].tolist()
    # 所有朝暮之印位置
    all_change_pos = set(tuple(x) for x in np.c_[(M0 == -1).nonzero()])
    result = []
    # 循环箱子数量小于等于4的所有放置方法
    for k in range(max_box+1):
        # 从所有可放置地块中挑选k个
        for pos in combinations(zero_pos, k):
            M = M0.copy()
            # 放置箱子
            for i, j in pos:
                M[i, j] = 1
            # 3个出口全部堵死则为非法状态，跳过
            if M[2, 6] > 0 and M[4, 6] > 0 and M[6, 6] > 0:
                continue
            illegal = False
            path_list = []
            for start in [0, 2, 4, 6]:
                # 计算每个怪物的路线
                path = find_path(M, (start, 0), (start, 7))
                # 路线为空则为非法状态
                if len(path) == 0:
                    illegal = True
                    break
                else:
                    path_list.append(path)
            if not illegal:
                # 所有怪物均可到达目标位置，记录箱子摆放方式与结果
                result.append({
                    'M': M,
                    'pos': pos,
                    'path_list': path_list,
                    'state': tuple([len(all_change_pos & set(path)) % 2 for path in path_list])
                })
    return result

def plot_path(M, path_list, state, subplot_args=None, **args):
    if subplot_args is not None:
        ax = plt.subplot(*subplot_args)
    else:
        ax = plt.subplot(1, 1, 1)
    cmap = colors.ListedColormap(['#FF3030', 'white', '#D2D2D2', 'white', '#00B400', 'black'])
    ax.matshow(M, cmap=cmap)
    for i, path in enumerate(path_list):
        ax.plot(
            [y + 0.1 * np.cos((i + 0.25) * np.pi / 2) for (x, y) in path], 
            [x + 0.1 * np.sin((i + 0.25) * np.pi / 2) for (x, y) in path],
            linewidth=2
        )
    ax.set_title(str(state), fontdict={'size':18})
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    if subplot_args is None:
        plt.show()
        
if __name__ == '__main__':
    '''
    初始地图，小于等于0代表可通行地块，大于0代表不可通行地块
    0代表可放置箱子地块，-1为朝暮之印所在地块
    -2为不考虑放置箱子地块，-3为出怪点
    1为箱子（可撤退），2为不可破坏地形
    注意到部分地块完全等价，如(H6,H7,G7)，(C7,B7,B8)等。对这部分地块，每组仅保留一个作为可放置地块，其余均置为-2
    '''
    M1 = np.array([
        [-3,  -2,  0, -1,  0,  -2,  2, -2],
        [ 0,   0,  2,  0,  2,  -2,  2, -2],
        [-3,   0,  0,  0,  0,  -1,  0, -2],
        [ 0,   0,  2,  0,  2,   0,  2,  0],
        [-3,   0,  0,  0,  0,  -1,  0, -2],
        [ 0,   0,  2,  0,  2,  -2,  2,  0],
        [-3,  -2,  0, -1,  2,  -2,  0, -2]
    ])

    M2 = np.array([
        [-3,  -2,  0, -1,  0,  -2,  2, -2],
        [ 0,   0,  2,  0,  2,  -2,  2, -2],
        [-3,   0,  0,  0,  2,  -1,  0, -2],
        [ 0,   0,  2,  0,  2,   0,  2,  0],
        [-3,   0,  2,  0,  0,  -1,  2, -2],
        [ 0,   0,  2,  0,  2,  -2,  2,  0],
        [-3,  -2,  0, -1,  2,  -2,  0, -2]
    ])
    result_1 = calculate_map(M1, 4)
    result_2 = calculate_map(M2, 4)
    
    state_dict_1 = {state: [] for state in product([0, 1], repeat=4)}
    for i, info in enumerate(result_1):
        state_dict_1[info['state']].append(i)
    plt.figure(figsize=(12, 4))
    for i, state in enumerate([(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 0, 1)]):
        i_tmp = minimal_by(state_dict_1[state], lambda i: len(result_1[i]['pos']))
        plot_path(**result_1[i_tmp], subplot_args=(1, 3, i+1))

    state_dict_2 = {state: [] for state in product([0, 1], repeat=4)}
    for i, info in enumerate(result_2):
        state_dict_2[info['state']].append(i)
    plt.figure(figsize=(12, 4))
    for i, state in enumerate([(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 0, 1)]):
        i_tmp = minimal_by(state_dict_2[state], lambda i: len(result_2[i]['pos']))
        plot_path(**result_2[i_tmp], subplot_args=(1, 3, i+1))
    plt.show()