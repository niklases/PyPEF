#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonian Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import ray
import warnings

from Modules_PLSR import Full_Path, Path_AAindex_Dir, XY, Get_R2

# to handle UserWarning for PLS n_components as error
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def Formatted_Output_Parallel(AAindex_R2_List, Minimum_R2=0.0):
    """
    takes the sorted list from function R2_List and writes the model names with an R2 ≥ 0
    as well as the corresponding number of components for each model so that the user gets
    a list (Model_Results.txt) of the top ranking models for the given validation set.
    """
    index, value, value2, value3, value4, n_com = [], [], [], [], [], []

    for (idx, val, val2, val3, val4, n_c) in AAindex_R2_List:
        if (val >= Minimum_R2):
            index.append(idx[:-4])
            value.append('{:f}'.format(val))
            value2.append('{:f}'.format(val2))
            value3.append('{:f}'.format(val3))
            value4.append('{:f}'.format(val4))
            n_com.append(n_c)

    if len(value) == 0:
        raise ValueError('No model with positive R2.')

    data = np.array([index, value, value2, value3, value4, n_com]).T
    col_width = max(len(str(value)) for row in data for value in row) + 5

    head = ['Index', 'R2', 'RMSE', 'NRMSE', 'Pearson_r', 'N_Components']
    with open('Model_Results.txt', 'w') as f:
        f.write("".join(caption.ljust(col_width) for caption in head) + '\n')
        f.write(len(head)*col_width*'-' + '\n')
        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')

    return ()


@ray.remote
def Parallel(d, Core, AAindices, Learning_Set, Validation_Set):
    """
    Parallelization of running using the user-defined number of cores.
    Defining the task for each core.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    AAindex_R2_List = []
    for i in range(d[Core][0], d[Core][1]):
        aaindex = AAindices[i]
        xy_learn = XY(Full_Path(aaindex), Learning_Set)
        x_learn, y_learn = xy_learn.Get_X_And_Y()

        # If x_learn (or y_learn) is an empty array, the sequence could not be encoded,
        # because of NoneType value. -> Skip

        if len(x_learn) != 0:
            xy_test = XY(Full_Path(aaindex), Validation_Set)
            x_test, y_test = xy_test.Get_X_And_Y()
            r2, rmse, nrmse, pearson_r, n_comp = Get_R2(x_learn, x_test, y_learn, y_test)
            AAindex_R2_List.append([aaindex, r2, rmse, nrmse, pearson_r, n_comp])

    return AAindex_R2_List


def R2_List_Parallel(Learning_Set, Validation_Set, Cores):
    """
    Parallelization of running using the user-defined number of cores.
    Calling function Parallel to execute the parallel running and
    getting the results from each core each being defined by a result ID.
    """
    AAindices = [file for file in os.listdir(Path_AAindex_Dir()) if file.endswith('.txt')]

    split = int(len(AAindices)/Cores)
    last_split = int(len(AAindices) % Cores) + split

    d = {}
    for i in range(Cores-1):
        d[i] = [i*split, i*split + split]
    d[Cores-1] = [(Cores-1)*split, (Cores-1)*split + last_split]

    result_ids = []
    for j in range(Cores):
        result_ids.append(Parallel.remote(d, j, AAindices, Learning_Set, Validation_Set))
    results = ray.get(result_ids)

    AAindex_R2_List = []
    for core in range(Cores):
        for j,_ in enumerate(results[core]):
            AAindex_R2_List.append(results[core][j])
    AAindex_R2_List.sort(key=lambda x: x[1], reverse=True)

    return AAindex_R2_List
