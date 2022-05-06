from func import *
import numpy as np
def probability(x_old, x_new, T):
    # 状态转移概率
    if x_new < x_old:
        return 1
    else:
        return np.exp(-(x_new - x_old) / T)


def temperature(t):
    # 冷却进度表
    return 50 / (1 + t)

def fitness_func_1(X):
    # 目标函数，即适应度值，X是种群的表现型
    x = X[:, 0]
    y = X[:, 1]
    return Goldstein_price(x, y)

def fitness_func_2(X):
    # 目标函数，即适应度值，X是种群的表现型
    x = X[:, 0]
    y = X[:, 1]
    return Rastrigin(x, y)

def decode(x, a, b):
    # 解码，即基因型到表现型
    xt = 0
    for i in range(len(x)):
        xt = xt + x[i] * np.power(2, i)
    return a + xt * (b - a) / (np.power(2, len(x)) - 1)

def decode_X(X: np.array):
    # 对整个种群的基因解码，上面的decode是对某个染色体的某个变量进行解码
    X2 = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        xi = decode(X[i, :10], -2, 2)
        yi = decode(X[i, 10:], -2, 2)
        X2[i, :] = np.array([xi, yi])
    return X2

def select(X, fitness):
    # 根据轮盘赌法选择优秀个体
    fitness = 1 / fitness  # fitness越小表示越优秀，被选中的概率越大，做 1/fitness 处理
    fitness = fitness / fitness.sum()  # 归一化
    idx = np.array(list(range(X.shape[0])))
    X2_idx = np.random.choice(idx, size=X.shape[0], p=fitness)  # 根据概率选择
    X2 = X[X2_idx, :]
    return X2

def crossover(X, c):
    # 按顺序选择2个个体以概率c进行交叉操作
    for i in range(0, X.shape[0], 2):
        xa = X[i, :]
        xb = X[i + 1, :]
        for j in range(X.shape[1]):
            # 产生0-1区间的均匀分布随机数，判断是否需要进行交叉替换
            if np.random.rand() <= c:
                xa[j], xb[j] = xb[j], xa[j]
        X[i, :] = xa
        X[i + 1, :] = xb
    return X

def mutation(X, m):
    # 变异操作
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() <= m:
                X[i, j] = (X[i, j] + 1) % 2
    return X



