#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import ray
import warnings

from Modules_Regression import Full_Path, Path_AAindex_Dir, XY, Get_R2

# to handle UserWarning for PLS n_components as error
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def Formatted_Output_Parallel(AAindex_R2_List, Minimum_R2=0.0, noFFT=False):
    """
    takes the sorted list from function R2_List and writes the model names with an R2 ≥ 0
    as well as the corresponding number of components for each model so that the user gets
    a list (Model_Results.txt) of the top ranking models for the given validation set.
    """
    index, value, value2, value3, value4, value5, regr_models, parameters = [], [], [], [], [], [], [], []

    for (idx, val, val2, val3, val4, val5, r_m, pam) in AAindex_R2_List:
        if (val >= Minimum_R2):
            index.append(idx[:-4])
            value.append('{:f}'.format(val))
            value2.append('{:f}'.format(val2))
            value3.append('{:f}'.format(val3))
            value4.append('{:f}'.format(val4))
            value5.append('{:f}'.format(val5))
            regr_models.append(r_m)
            parameters.append(pam)

    if len(value) == 0:
        raise ValueError('No model with positive R2.')

    data = np.array([index, value, value2, value3, value4, value5, regr_models, parameters]).T
    col_width = max(len(str(value)) for row in data for value in row[:-1]) + 5

    head = ['Index', 'R2', 'RMSE', 'NRMSE', 'Pearson_r', 'Regression', 'Model parameters']
    with open('Model_Results.txt', 'w') as f:
        if noFFT is not False:
            f.write("No FFT used in this model construction, performance"
                    " represents model accuracies on raw encoded sequence data.\n\n")

        heading = "".join(caption.ljust(col_width) for caption in head) + '\n'
        f.write(heading)

        row_length = []
        for row in data:
            row_ = "".join(str(value).ljust(col_width) for value in row) + '\n'
            row_length.append(len(row_))
        row_length_max = max(row_length)
        f.write(row_length_max * '-' + '\n')

        for row in data:
            f.write("".join(str(value).ljust(col_width) for value in row) + '\n')

    return ()


@ray.remote
def Parallel(d, Core, AAindices, Learning_Set, Validation_Set, regressor='pls', noFFT=False):
    """
    Parallelization of running using the user-defined number of cores.
    Defining the task for each core.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    AAindex_R2_List = []
    for i in range(d[Core][0], d[Core][1]):
        aaindex = AAindices[i]  # Parallelization of AAindex iteration
        xy_learn = XY(Full_Path(aaindex), Learning_Set)

        if noFFT == False:  # X is FFT-ed of encoded alphabetical sequence
            x_learn, y_learn, _ = xy_learn.Get_X_And_Y()
        else:  # X is raw encoded of alphabetical sequence
            _, y_learn, x_learn = xy_learn.Get_X_And_Y()

        # If x_learn (or y_learn) is an empty array, the sequence could not be encoded,
        # because of NoneType value. -> Skip
        if len(x_learn) != 0:
            xy_test = XY(Full_Path(aaindex), Validation_Set)

            if noFFT == False:  # X is FFT-ed of the encoded alphabetical sequence
                x_test, y_test, _ = xy_test.Get_X_And_Y()
            else:  # X is the raw encoded of alphabetical sequence
                _, y_test, x_test = xy_test.Get_X_And_Y()

            r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params = Get_R2(x_learn, x_test, y_learn, y_test,
                                                                                      regressor)
            AAindex_R2_List.append([aaindex, r2, rmse, nrmse, pearson_r, spearman_rho, regressor, best_params])

    return AAindex_R2_List


def R2_List_Parallel(Learning_Set, Validation_Set, Cores, regressor='pls', noFFT=False, sort='1'):
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
    for j in range(Cores):  # Parallel running
        result_ids.append(Parallel.remote(d, j, AAindices, Learning_Set, Validation_Set, regressor, noFFT))

    results = ray.get(result_ids)

    AAindex_R2_List = []
    for core in range(Cores):
        for j, _ in enumerate(results[core]):
            AAindex_R2_List.append(results[core][j])

    try:
        sort = int(sort)
        if sort == 2 or sort == 3:
            AAindex_R2_List.sort(key=lambda x: x[sort])
        else:
            AAindex_R2_List.sort(key=lambda x: x[sort], reverse=True)

    except ValueError:
        raise ValueError("Choose between options 1 to 5 (R2, RMSE, NRMSE, Pearson's r, Spearman's rho.")

    return AAindex_R2_List