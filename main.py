import numpy as np
from pdb import set_trace

##########################
# Eric Vargas[]
# NUSP 2370310
##########################

#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
import os
from xlib.elliptic import f_example  # run __init__.py
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--n', type=int, help='Ver a apresentacao do Projeto')

    args = parser.parse_args()

    #err = f_example(4)
    n = np.arange(3, args.n)
    for i in n:
        err = f_example(i)
        #print(f"{i}: {err}")
        print(f"{i}: \t{1/i**4} \t{err}")
