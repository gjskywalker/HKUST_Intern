from numpy import *


def Goldstein_price(x1, x2):
    f = (1 + pow((x1 + x2 + 1), 2) *
         (19 - 14 * x1 + 3 * pow(x1, 2) - 14 * x2 + 6 * x2 * x1 + 3 * pow(x2, 2))) * (
                30 + pow((2 * x1 - 3 * x2), 2) * (
                18 - 32 * x1 + 12 * pow(x1, 2) + 48 * x2 - 36 * x1 * x2 + 27 * pow(x2, 2)))
    return f


def Rastrigin(x1, x2):
    f = 10 * 2 + (pow(x1, 2) - 10 * cos(2 * pi * x1)) + (pow(x2, 2) - 10 * cos(2 * pi * x2))
    return f


if __name__ == '__main__':
    print(1)
