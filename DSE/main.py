from DSE import *
import argparse

parser = argparse.ArgumentParser(description='algorithm_test :all the result will be saved in log')
parser.add_argument('--dse', type=str, help='algorithm in DSE.py | Grid_search | Greedy | Naive_search | '
                                            'Random_search | Simulated_Annealing | Genetic_Algorithm')
parser.add_argument('--func', type=str, help='func in func.py | gp:Goldstein_price | ra:Rastrigin')
args = parser.parse_args()

if args.dse == "Grid_search":
    Grid_search(args.func)
elif args.dse == "Greedy":
    Greedy(args.func)
elif args.dse == "Naive_search":
    Naive_search(args.func)
elif args.dse == "Random_search":
    Random_search(args.func)
elif args.dse == "Simulated_Annealing":
    Simulated_Annealing(args.func)
elif args.dse == "Genetic_Algorithm":
    Genetic_Algorithm(args.func)
