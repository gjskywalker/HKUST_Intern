from func import *
import numpy as np
import logging
from timeit import default_timer as timer
import random
from utils import *

def Genetic_Algorithm(func):
    # 遗传算法主函数
    if func == 'gp':
        logging.basicConfig(filename="log/Goldstein_price.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        c = 0.3  # 交叉概率
        m = 0.05  # 变异概率
        best_fitness = []  # 记录每次迭代的效果
        best_xy = []  # 最佳的x1,x2值
        iter_num = 100  # 最大迭代次数
        X0 = np.random.randint(-2, 2, (50, 40))  # 随机初始化种群，为50*40的0-1矩阵
        for i in range(iter_num):
            X1 = decode_X(X0)  # 染色体解码
            fitness = fitness_func_1(X1)  # 计算个体适应度
            X2 = select(X0, fitness)  # 选择操作
            X3 = crossover(X2, c)  # 交叉操作
            X4 = mutation(X3, m)  # 变异操作
            # 计算一轮迭代的效果
            X5 = decode_X(X4)
            fitness = fitness_func_1(X5)
            best_fitness.append(fitness.min())
            x, y = X5[fitness.argmin()]
            best_xy.append((x, y))
            X0 = X4
        toc = timer()
        # 多次迭代后的最终效果
        logging.info('Genetic_Algorithm')
        logging.info('Minimum f: %.4f' % best_fitness[-1])
        logging.info('Optimum x1: %f' % best_xy[-1][0])
        logging.info('Optimum x2: %f' % best_xy[-1][1])
        logging.info('Time : %fs\n' % (toc - tic))
    if func == 'ra':
        logging.basicConfig(filename="log/Rastrigin.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        c = 0.3  # 交叉概率
        m = 0.05  # 变异概率
        best_fitness = []  # 记录每次迭代的效果
        best_xy = []
        iter_num = 100  # 最大迭代次数
        X0 = np.random.randint(-5, 5, (50, 40))  # 随机初始化种群，为50*40的0-1矩阵
        for i in range(iter_num):
            X1 = decode_X(X0)  # 染色体解码
            fitness = fitness_func_2(X1)  # 计算个体适应度
            X2 = select(X0, fitness)  # 选择操作
            X3 = crossover(X2, c)  # 交叉操作
            X4 = mutation(X3, m)  # 变异操作
            # 计算一轮迭代的效果
            X5 = decode_X(X4)
            fitness = fitness_func_2(X5)
            best_fitness.append(fitness.min())
            x, y = X5[fitness.argmin()]
            best_xy.append((x, y))
            X0 = X4
        toc = timer()
        # 多次迭代后的最终效果
        logging.info('Genetic_Algorithm')
        logging.info('Minimum f: %.4f' % best_fitness[-1])
        logging.info('Optimum x1: %f' % best_xy[-1][0])
        logging.info('Optimum x2: %f' % best_xy[-1][1])
        logging.info('Time : %fs\n' % (toc - tic))
    return -1

def Random_search(func):
    if func == 'gp':
        tic = timer()
        logging.basicConfig(filename="log/Goldstein_price.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        x_min, y_min, min_value = 0, 0, float('inf')
        for i in range(10000):
            tmp_x, tmp_y = round(random.uniform(-2, 2), 2), round(random.uniform(-2, 2), 2)
            tmp_value = Goldstein_price(tmp_x, tmp_y)
            if tmp_value < min_value:
                x_min = tmp_x
                y_min = tmp_y
                min_value = tmp_value
        toc = timer()
        logging.info('Random Search')
        logging.info('Minimum f: %.4f' % min_value)
        logging.info('Optimum x1: %f' % x_min)
        logging.info('Optimum x2: %f' % y_min)
        logging.info('Time : %fs\n' % (toc - tic))
    if func == 'ra':
        tic = timer()
        logging.basicConfig(filename="log/Rastrigin.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        x_min, y_min, min_value = 0, 0, float('inf')
        for i in range(10000):
            tmp_x, tmp_y = round(random.uniform(-5, 5), 2), round(random.uniform(-5, 5), 2)
            tmp_value = Rastrigin(tmp_x, tmp_y)
            if tmp_value < min_value:
                x_min = tmp_x
                y_min = tmp_y
                min_value = tmp_value
        toc = timer()
        logging.info('Random Search')
        logging.info('Minimum f: %.4f' % min_value)
        logging.info('Optimum x1: %f' % x_min)
        logging.info('Optimum x2: %f' % y_min)
        logging.info('Time : %fs\n' % (toc - tic))
    return -1

def Grid_search(func):
    if func == 'gp':
        logging.basicConfig(filename="log/Goldstein_price.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        graph_x = []
        graph_y = []
        graph_z = []
        for x1 in np.arange(-2.0, 2.001, 0.001):
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            for x2 in np.arange(-2.0, 2.001, 0.001):
                f = Goldstein_price(x1, x2)
                graph_x_row.append(x1)
                graph_y_row.append(x2)
                graph_z_row.append(f)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)  # 这里是一个组网过程，就是将x1,x2,f的值进行组网，
            # 其实如果只需要求得最小值主要对f值进行组网，但是由于需要得到x1,x2的值所以将x1和x2也加入了其中
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)  # 用np的内置函数求出最小值
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        toc = timer()
        logging.info('Grid Search')
        logging.info('Minimum f: %.4f' % min_z)
        logging.info('Optimum x1: %f' % (graph_x[pos_min_z[0], pos_min_z[1]]))
        logging.info('Optimum x2: %f' % (graph_y[pos_min_z[0], pos_min_z[1]]))
        logging.info('Time : %fs\n' % (toc - tic))
    if func == 'ra':
        logging.basicConfig(filename="log/Rastrigin.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        graph_x = []
        graph_y = []
        graph_z = []
        for x1 in np.arange(-5.01, 5.01, 0.01):
            graph_x_row = []
            graph_y_row = []
            graph_z_row = []
            for x2 in np.arange(-5.01, 5.01, 0.01):
                f = Rastrigin(x1, x2)
                graph_x_row.append(x1)
                graph_y_row.append(x2)
                graph_z_row.append(f)
            graph_x.append(graph_x_row)
            graph_y.append(graph_y_row)
            graph_z.append(graph_z_row)
        graph_x = np.array(graph_x)
        graph_y = np.array(graph_y)
        graph_z = np.array(graph_z)
        min_z = np.min(graph_z)
        pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
        toc = timer()
        logging.info('Grid Search')
        logging.info('Minimum f: %.4f' % min_z)
        logging.info('Optimum x1: %f' % (graph_x[pos_min_z[0], pos_min_z[1]]))
        logging.info('Optimum x2: %f' % (graph_y[pos_min_z[0], pos_min_z[1]]))
        logging.info('Time : %fs\n' % (toc - tic))
    return -1

def Greedy(func):
    if func == 'gp':
        logging.basicConfig(filename="log/Goldstein_price.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        step = 0.01
        best_sol = [round(random.uniform(-2, 2), 2),
                    round(random.uniform(-2, 2), 2)]  # 最佳解x1、x2写为数组形式， round()保留至两位小数， 因为取step为0.01

        while True:
            tmp_min_value = Goldstein_price(best_sol[0], best_sol[1])  # tmp_min_val记录每次Greedy搜索前，也即当前位置的函数值
            min_value = tmp_min_value  # min_val用于记录Greedy搜索过程中的最小值，初始值为当前位置函数值，搜索过程中可能变也可能不变
            tmp_sol = []  # tmp_sol用于记录Greedy搜索范围内的每组[x1,x2]

            for i in range(2):  # 因为有x1和x2两个参数， 所以用for循环更新两次， 每次分别更新x1和x2
                if best_sol[i] > -2:  # 如果x1/x2的值未超过下边界
                    tmp_sol.append(best_sol[0:i] + [best_sol[i] - step] + best_sol[i + 1:])  # 将x1/x2的值-0.01，其它值不变
                if best_sol[i] < 2:
                    tmp_sol.append(best_sol[0:i] + [best_sol[i] + step] + best_sol[i + 1:])

            # for循环结束后, tmp_sol中存储了[x1-0.01,x2], [x1+0.01,x2], [x1,x2-0.01], [x1,x2+0.01]四组值

            for sol in tmp_sol:
                tmp_value = Goldstein_price(sol[0], sol[1])
                if tmp_value < min_value:  # 如果出现新的最小值，更新最佳解和最小值
                    best_sol = sol
                    min_value = tmp_value

            if min_value == tmp_min_value:  # 如果min_value未更新，说明当前解已为局部最优，break
                break
        toc = timer()
        logging.info('Greedy')
        logging.info('Minimum f: %.4f' % min_value)
        logging.info('Optimum x1: %f' % best_sol[0])
        logging.info('Optimum x2: %f' % best_sol[1])
        logging.info('Time : %fs\n' % (toc - tic))
    if func == 'ra':
        logging.basicConfig(filename="log/Rastrigin.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        step = 0.01
        best_sol = [round(random.uniform(-5, 5), 2),
                    round(random.uniform(-5, 5), 2)]

        while True:
            tmp_min_value = Rastrigin(best_sol[0], best_sol[1])
            min_value = tmp_min_value
            tmp_sol = []

            for i in range(2):
                if best_sol[i] > -5:
                    tmp_sol.append(best_sol[0:i] + [best_sol[i] - step] + best_sol[i + 1:])
                if best_sol[i] < 5:
                    tmp_sol.append(best_sol[0:i] + [best_sol[i] + step] + best_sol[i + 1:])

            for sol in tmp_sol:
                tmp_value = Rastrigin(sol[0], sol[1])
                if tmp_value < min_value:
                    best_sol = sol
                    min_value = tmp_value

            if min_value == tmp_min_value:
                break
        toc = timer()
        logging.info('Greedy')
        logging.info('Minimum f: %.4f' % min_value)
        logging.info('Optimum x1: %f' % best_sol[0])
        logging.info('Optimum x2: %f' % best_sol[1])
        logging.info('Time : %fs\n' % (toc - tic))
    return -1

def Naive_search(func):
    if func == 'gp':
        logging.basicConfig(filename="log/Goldstein_price.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        x1 = x2 = 0
        tic = timer()
        tmp2 = 100
        for i in np.arange(-2, 2.01, 0.01):
            for j in np.arange(-2, 2.01, 0.01):
                tmp1 = Goldstein_price(i, j)
                if tmp2 > tmp1:
                    tmp2 = tmp1
                    x1 = i
                    x2 = j
        toc = timer()
        logging.info('naive_search')
        logging.info('Minimum f: %.4f' % tmp2)
        logging.info('Optimum x1: %f' % x1)
        logging.info('Optimum x2: %f' % x2)
        logging.info('Time : %fs\n' % (toc - tic))
    if func == 'ra':
        logging.basicConfig(filename="log/Rastrigin.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        x1 = x2 = 0
        tic = timer()
        tmp2 = 100
        for i in np.arange(-5, 5.01, 0.01):
            for j in np.arange(-5, 5.01, 0.01):
                tmp1 = Rastrigin(i, j)
                if tmp2 > tmp1:
                    tmp2 = tmp1
                    x1 = i
                    x2 = j
        toc = timer()
        logging.info('naive_search')
        logging.info('Minimum f: %.4f' % tmp2)
        logging.info('Optimum x1: %f' % x1)
        logging.info('Optimum x2: %f' % x2)
        logging.info('Time : %fs\n' % (toc - tic))
    return -1

def Simulated_Annealing(func):
    if func == 'gp':
        logging.basicConfig(filename="log/Goldstein_price.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        state = [round(random.uniform(-2, 2), 2),
                 round(random.uniform(-2, 2), 2)]
        x1, x2 = state[0], state[1]
        e = Goldstein_price(x1, x2)
        bestState = state
        bestE = e
        for iter in range(0, 5000):
            delta = np.random.rand(2) - 0.5
            nextState = state + delta * 1.
            nextE = Goldstein_price(nextState[0], nextState[1])

            if bestE > nextE:
                bestState = nextState
                bestE = nextE
            r = np.random.rand()
            if probability(e, nextE, temperature(iter / 100)) >= r:
                state = nextState
                e = nextE
        toc = timer()
        logging.info('Simulated_Annealing')
        logging.info('Minimum f: %.4f' % bestE)
        logging.info('Optimum x1: %f' % bestState[0])
        logging.info('Optimum x2: %f' % bestState[1])
        logging.info('Time : %fs\n' % (toc - tic))
    if func == 'ra':
        logging.basicConfig(filename="log/Rastrigin.log", level=logging.INFO, filemode="a",
                            format="%(asctime)s|%(message)s")
        tic = timer()
        state = [round(random.uniform(-5, 5), 2),
                 round(random.uniform(-5, 5), 2)]
        x1, x2 = state[0], state[1]
        e = Rastrigin(x1, x2)
        bestState = state
        bestE = e
        for iter in range(0, 40000):
            delta = np.random.rand(2) - 0.5
            nextState = state + delta * 1.
            nextE = Rastrigin(nextState[0], nextState[1])

            if bestE > nextE:
                bestState = nextState
                bestE = nextE
            r = np.random.rand()
            if probability(e, nextE, temperature(iter / 100)) >= r:
                state = nextState
                e = nextE
        toc = timer()
        logging.info('Simulated_Annealing')
        logging.info('Minimum f: %.4f' % bestE)
        logging.info('Optimum x1: %f' % bestState[0])
        logging.info('Optimum x2: %f' % bestState[1])
        logging.info('Time : %fs\n' % (toc - tic))
    return -1


if __name__ == '__main__':
    print(1)
