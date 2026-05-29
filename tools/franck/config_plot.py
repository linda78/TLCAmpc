# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 10:13:43 2026

@author: Hamekong
"""

# %% imports

import matplotlib.pyplot as plt
from distutils.spawn import find_executable

#%% configurations
def configure_plt(check_latex = True):
    """
    Set Font sizes for plots.
    
    Arg:
        check_latex : bool, optional
            Use LaTex-mode (if available). The default is True.
    """

    
    if check_latex:
        
        if find_executable('latex'):
            plt.rc('text', usetex=True)
        else:
            plt.rc('text', usetex=False)
    plt.rc('font',family='Times New Roman')
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    small_size = 13
    small_medium = 14
    medium_size = 16
    big_medium = 20
    big_size = 22
        
    plt.rc('font', size = small_size)          # controls default text sizes
    plt.rc('axes', titlesize = big_size)     # fontsize of the axes title
    plt.rc('axes', labelsize = big_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = big_medium)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = big_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize = small_size)    # legend fontsize
    plt.rc('figure', titlesize = big_size)  # fontsize of the figure title
    
    plt.rc('grid', c='0.5', ls='-', lw=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.close()
    
# %% test

if __name__ == '__main__':
    configure_plt()