# DSE Algorithm Comparison

- ### Test Function and Algorithm Principles

This part of work is stored in **the PDF document**, containing test function, introduction of the DSE algorithm, results, and obtained specific parameter settings.

- ### Project Introduction

  - DSE.py includes six algorithms
  - func.py includes two test functions 
  - utils.py includes six functions for SDE algorithm
  - plot.py used to show the figure of two test functions

- ### Algorithm Test

 The instructions are as follows:

```shell
cd DSE
pip install numpy
pip install matploylib
```

  ```python
  usage: main.py [-h] [--dse DSE] [--func FUNC]
  
  algorithm_test :all the result will be saved in log	
  
  optional arguments:
    -h, --help   show this help message and exit
    --dse DSE    algorithm in DSE.py | Grid_search | Greedy | Naive_search | Random_search | Simulated_Annealing | Genetic_Algorithm
    --func FUNC  func in func.py | gp:Goldstein_price | ra:Rastrigin
  
  example:
  python main.py --dse=Grid_search --func=gp
  ```

- ### Test Result

  Store two test results of both functions in “log”, the formats are as follows

  ```
  2022-05-05 12:26:43,651|Random Search
  2022-05-05 12:26:43,651|Minimum f: 0.8839
  2022-05-05 12:26:43,651|Optimum x1: -0.060000
  2022-05-05 12:26:43,651|Optimum x2: 0.030000
  2022-05-05 12:26:43,651|Time : 0.044655s
  ```

- ### Algorithm Test Comparision

  |                 | Greedy_search | Simulated_Annealing | Random_search | Genetic_Algorithm | Naive_search | Grid_search |
  | --------------- | ------------- | ------------------- | ------------- | ----------------- | ------------ | ----------- |
  | Goldstein_price | 0.006260 s    | 0.061014 s          | 0.079004 s    | 0.699202 s        | 0.994265 s   | 1.062211 s  |
  | Rastrigin       | 0.010594 s    | 0.375557 s          | 0.079583 s    | 0.700793  s       | 3.723845 s   | 4.175957 s  |

