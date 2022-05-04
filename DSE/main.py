import logging
from func import *
import numpy as np

logging.basicConfig(filename="Rastrigin.log", level=logging.INFO, filemode="w",
                    format="%(asctime)s|%(message)s")

tmp2 = 100
for i in np.arange(-5, 5.001, 0.001):
    for j in np.arange(-5, 5.001, 0.001):
        tmp1 = Rastrigin(i, j)
        if tmp2 > tmp1:
            tmp2 = tmp1
            logging.info("f=%f x1=%f x2=%f", tmp2, i, j)
