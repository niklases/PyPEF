#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

# PyPEF – An Integrated Framework for Data-Driven Protein Engineering
# Journal of Chemical Information and Modeling, 2021, 61, 3463-3476
# https://doi.org/10.1021/acs.jcim.1c00099
# Niklas E. Siedhoff1,§, Alexander-Maurice Illig1,§, Ulrich Schwaneberg1,2, Mehdi D. Davari1,*
# 1Institute of Biotechnology, RWTH Aachen University, Worringer Weg 3, 52074 Aachen, Germany
# 2DWI-Leibniz Institute for Interactive Materials, Forckenbeckstraße 50, 52074 Aachen, Germany
# *Corresponding author
# §Equal contribution


from itertools import chain
import random
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import ray

from pypef.aaidx.cli.regression import (
    cv_regression_options, count_mutation_levels_and_get_dfs, process_df_encoding
)
from pypef.dca.model import DCAHybridModel


def split_train_sizes_for_multiprocessing(train_sizes, n_cores):
    sum_training_sizes = sum(train_sizes)
    sum_single = 0
    train_splits, total_train_splits = [], []
    for single in train_sizes:
        sum_single += single
        if sum_single <= sum_training_sizes // n_cores:
            train_splits.append(single)
        else:
            train_splits.append(single)
            total_train_splits.append(train_splits)
            train_splits = []
            sum_single = 0
    len_train_splits = len(list(chain.from_iterable(total_train_splits)))
    remaining_train_sizes = train_sizes[len_train_splits:]
    for remain in remaining_train_sizes:
        total_train_splits.append([remain])
    total_train_splits = [x for x in total_train_splits if x != []]
    while len(total_train_splits) < n_cores:
        split_1 = total_train_splits[0][:int(len(total_train_splits[0]) * 3 / 4)]
        split_2 = total_train_splits[0][int(len(total_train_splits[0]) * 3 / 4):]
        total_train_splits[0] = split_1
        total_train_splits.insert(1, split_2)
    while len(total_train_splits) > n_cores:
        total_train_splits[0] = total_train_splits[0] + total_train_splits[1]
        total_train_splits.pop(1)

    return total_train_splits


def plot(
        data: dict,
        dataset: str,
        n_runs: int = 10,
        mut_extrapol: bool = False,
        conc: bool = False
):
    """
    Description
    ----------
    Plots mutation extrapolation performances (measured
    vs. predicted fitness values of variants) at different levels/degrees
    of substitutions, e.g. trained on single substituional variants
    and predicting double, triple, and higher substituted variants.
    Further, 'conc' allows to learn on concatenated levels of substitutions,
    e.g. fist learning on single and predicting double, then learning on
    single+double and predicting triple, then learning on single+double+triple
    and predicting quadruple substituted variants, and so on.


    Parameters
    ----------

    Returns
    ----------
    None
        Just plots the performances.
    """

    def get_mean_and_std(
        x: list,
        sort_arr: np.ndarray,
    ) -> tuple:
        x = [s for _, s in sorted(zip(sort_arr, x))]
        x = np.array(x)
        x = np.split(np.array(x), int(len(x) / n_runs))
        x = np.vstack(x)
        return np.mean(x, axis=1), np.std(x, axis=1, ddof=1, dtype=float)

    training_sizes, testing_sizes, spearmanrs, \
        beta_1s, beta_2s, alphas, lvl = [], [], [], [], [], [], []
    for key in data.keys():
        print(key)
        training_sizes.append(data[key]['n_y_train'])
        testing_sizes.append(data[key]['n_y_test'])
        spearmanrs.append(data[key]['spearman_rho'])
        beta_1s.append(data[key]['beta_1'])
        beta_2s.append(data[key]['beta_2'])
        alphas.append(data[key]['alpha'])
        if mut_extrapol:
            if not conc:
                lvl.append(data[key]['test_mut_level'])
            else:
                lvl.append(data[key]['conc_test_mut_level'])
    for k in [training_sizes, testing_sizes, spearmanrs, beta_1s, beta_2s, alphas, lvl]:
        print(k)
    sort_arr = np.argsort(training_sizes)

    x_training, _ = get_mean_and_std(training_sizes, sort_arr)
    x_testing, _ = get_mean_and_std(testing_sizes, sort_arr)

    srs, srs_yerr = get_mean_and_std(spearmanrs, sort_arr)

    fig, ax = plt.subplots()

    if not mut_extrapol:
        ax.scatter(x_training, srs, marker='x')
        ax.errorbar(x_training, srs, yerr=srs_yerr, linestyle='', capsize=3, color='gray')
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Number of datapoints in the training set', size=12)

        ax2 = ax.twiny()
        ax2.scatter(x_training / (x_testing + x_training), srs, marker='')
        ax2.set_xlabel('Relative size of the training set', size=12)

    else:
        ax.scatter(lvl, srs, marker='x')
        ax.set_xticks([item for item in lvl], labels=[item for item in lvl])
        ax.set_xlabel('Mutational lvl of test set', size=12)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[-1] = 'All'
        ax.set_xticklabels(labels)

        for l, s, tr, te in zip(lvl, srs, training_sizes, testing_sizes):
            ax.annotate(f'N_train = {tr}\nN_test = {te}', xy=(l, s), textcoords='offset points', size=4)

        # if conc:
        #    ax2.scatter(lvl_conc, srs_conc, marker='')

    ax.set_ylabel(r"Spearman's $\rho$", size=12)
    plt.savefig('%s.png' % dataset, dpi=500, bbox_inches='tight')


@ray.remote
def _low_n_parallel_hybrid(hybrid_model: DCAHybridModel, train_sizes):
    return hybrid_model.run(train_sizes=train_sizes)


def low_n_hybrid_model(
        hybrid_model: DCAHybridModel,
        n_cores: int,
        out_files: str = 'out',
        n_runs: int = 10
):
    train_sizes = hybrid_model.get_train_sizes()
    if n_cores != 1:  # hyperthreading
        if len(train_sizes) < n_cores:
            n_cores = len(train_sizes)
        train_size_splits = split_train_sizes_for_multiprocessing(train_sizes, n_cores)
        results = ray.get([
            _low_n_parallel_hybrid.remote(
                hybrid_model,
                train_size_splits[i]) for i in range(len(train_size_splits))
        ])
        data = {}
        for r in results:
            data.update(r)
    else:
        data = {}
        for train_size in tqdm(train_sizes):
            data.update(hybrid_model.run(list(np.atleast_1d(train_size))))
            print('len data sc', len(data))

    np.save(f'{out_files}_hybrid_model_data.npy', data)
    plot(data, f'{out_files}', n_runs=n_runs)


def get_train_sizes(number_variants) -> np.ndarray:
    """
    Generates a list of train sizes to perform low-n with.
    Returns
    -------
    Numpy array of train sizes up to 80% (i.e. 0.8 * N_variants).
    """
    eighty_percent = int(number_variants) * 0.8
    train_sizes = np.sort(np.concatenate([
        np.arange(15, 50, 5), np.arange(50, 100, 10),
        np.arange(100, 150, 20), [160, 200, 250, 300, eighty_percent],
        np.arange(400, 1100, 100)
    ]))
    idx_max = np.where(train_sizes >= eighty_percent)[0][0] + 1

    return train_sizes[:idx_max]


def plot_low_n(
        train_sizes,
        avg_spearmanr,
        stddev_spearmanr,
        plt_name: str = ''
):
    plt.plot(train_sizes, avg_spearmanr, 'ko--', linewidth=1, markersize=1.5)
    plt.fill_between(
        np.array(train_sizes),
        np.array(avg_spearmanr) + np.array(stddev_spearmanr),
        np.array(avg_spearmanr) - np.array(stddev_spearmanr),
        alpha=0.5
    )
    plt.xlabel('Train sizes')
    plt.ylabel(r"Spearman's $\rho$")
    plt.savefig(plt_name + '.png', dpi=500)


def low_n(
        encoded_csv: str,
        cv_regressor: str = None,
        n_runs: int = 10,
        hybrid_modeling: bool = False,
        train_size_train: float = 0.66
):
    df = pd.read_csv(encoded_csv, sep=';', comment='#')
    if cv_regressor:
        name = 'ml_' + cv_regressor
        if cv_regressor == 'pls_loocv':
            raise SystemError('PLS LOOCV is not (yet) implemented '
                              'for the extrapolation task. Please choose'
                              'another CV regressor.')
        regressor = cv_regression_options(cv_regressor)
    elif hybrid_modeling:
        name = 'hybrid_ridge'
    n_variants = df.shape[0]
    train_sizes = get_train_sizes(n_variants)
    variants, X, y = process_df_encoding(df)
    avg_spearmanr, stddev_spearmanr = [], []
    # test_sizes = [n_variants - size for size in train_sizes]
    for size in tqdm(train_sizes):
        spearmanr_nruns = []
        for _ in range(n_runs):
            train_idxs = random.sample(range(n_variants - 1), int(size))
            test_idxs = []
            for n in range(n_variants - 1):
                if n not in train_idxs:
                    test_idxs.append(n)
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_test, y_test = X[test_idxs], y[test_idxs]

            if hybrid_modeling:
                x_wt = X[0]  # WT should be first CSV variant entry
                hybrid_model = DCAHybridModel(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=None,
                    y_test=None,
                    X_wt=x_wt
                )
                beta_1, beta_2, reg = hybrid_model.settings(
                    X_train, y_train, train_size_train=train_size_train)
                spearmanr_nruns.append(
                    hybrid_model.spearmanr(
                        y_test,
                        hybrid_model.predict(X_test, reg, beta_1, beta_2)
                    )
                )

            else:  # ML
                regressor.fit(X_train, y_train)
                # Best CV params: best_params = regressor.best_params_
                y_pred = regressor.predict(X_test)
                spearmanr_nruns.append(stats.spearmanr(y_test, y_pred)[0])

        avg_spearmanr.append(np.mean(spearmanr_nruns))
        stddev_spearmanr.append(np.std(spearmanr_nruns, ddof=1))

    plot_low_n(
        train_sizes,
        avg_spearmanr,
        stddev_spearmanr,
        'low_N_' + str(encoded_csv[:-4] + '_' + name)
    )

    return train_sizes, avg_spearmanr, stddev_spearmanr


def performance_mutation_extrapolation(
        encoded_csv: str,
        cv_regressor: str = None,
        train_size: float = 0.66,
        conc: bool = False,
        save_model: bool = True,
        hybrid_modeling: bool = False
) -> tuple:
    df = pd.read_csv(encoded_csv, sep=';', comment='#')
    if cv_regressor:
        if cv_regressor == 'pls_loocv':
            raise SystemError('PLS LOOCV is not (yet) implemented '
                              'for the extrapolation task. Please choose'
                              'another CV regressor.')
        regressor = cv_regression_options(cv_regressor)
        beta_1, beta_2 = None, None
    else:
        regressor = None
    data, data_conc = {}, {}
    # run extrapolation // implement train_size and n_runs?
    collected_levels = []
    for i_m, mutation_level_df in enumerate(count_mutation_levels_and_get_dfs(df)):
        if mutation_level_df.shape[0] != 0:
            collected_levels.append(i_m)
    train_idx_appended = []
    # train_df_appended = pd.DataFrame()
    if len(collected_levels) > 1:
        train_idx = collected_levels[0]
        train_df = df[train_idx]
        train_variants, X_train, y_train = process_df_encoding(train_df)
        if hybrid_modeling:
            x_wt = train_variants[0]
            hybrid_model = DCAHybridModel(
                X_train=X_train,
                y_train=y_train,
                X_test=None,
                y_test=None,
                X_wt=x_wt
            )
            beta_1, beta_2, reg = hybrid_model.settings(
                X_train, y_train, train_size_train=train_size)
            if save_model:
                print(f'Save model as... HYBRID_LVL_1.pkl')
                pickle.dump(reg, open('Pickles/HYBRID_LVL_1', 'wb'))  # Pickles/ not there construction try os.mkdir
        elif cv_regressor:
            regressor.fit(X_train, y_train)
            if save_model:
                print(f'Save model as... HYBRID_LVL_1.pkl')
                pickle.dump(regressor, open('Pickles/ML_LVL_1', 'wb'))  # Pickles/ not there construction try os.mkdir
        for i, _ in enumerate(collected_levels):
            if i < len(collected_levels) - 1:  # not last i for training
                print(i)
                # For training on distinct iterated level i, uncomment:
                # train_idx = collected_levels[i]
                # train_df = self.mutation_level_dfs[train_idx]
                # train_variants, X_train, y_train = self._process_df_encoding(train_df)
                test_idx = collected_levels[i + 1]
                test_df = df[test_idx]
                test_variants, X_test, y_test = process_df_encoding(test_df)
                if hybrid_model:
                    if beta_2 == 0.0:
                        alpha = np.nan
                    else:
                        alpha = reg.alpha
                    data.update(
                        {f'single_level_{test_idx + 1}_size_{len(y_test)}':
                            {
                                'test_mut_level': test_idx + 1,
                                'n_y_train': len(y_train),
                                'n_y_test': len(y_test),
                                'spearman_rho': hybrid_model.spearmanr(
                                    y_test,
                                    hybrid_model.predict(X_test, reg, beta_1, beta_2)
                                ),
                                'beta_1': beta_1,
                                'beta_2': beta_2,
                                'alpha': alpha
                            }
                        }
                    )
                else:  # ML
                    data.update(
                        {f'single_level_{test_idx + 1}_size_{len(y_test)}':
                            {
                                'test_mut_level': test_idx + 1,
                                'n_y_train': len(y_train),
                                'n_y_test': len(y_test),
                                'spearman_rho': stats.spearmanr(
                                    y_test,                    # Call predict on the BaseSearchCV estimator
                                    regressor.predict(X_test)  # with the best found parameters
                                )[0]
                            }
                        }
                    )

                if conc:  # training on mutational levels i: 1, ..., max(i)-1
                    print('CONC MODE')
                    train_idx_appended.append(collected_levels[i])
                    print(train_idx_appended)
                    if i != 0 and i < len(collected_levels) - 1:  # not the last (all_higher)
                        train_df_appended_conc = pd.DataFrame()
                        for idx in train_idx_appended:
                            train_df_appended_conc = pd.concat(
                                [train_df_appended_conc, df[idx]])
                        train_variants_conc, X_train_conc, y_train_conc = \
                            process_df_encoding(train_df_appended_conc)
                        print(len(X_train_conc) + len(y_test))
                        if hybrid_model:  # updating hybrid model params with newly inputted concatenated train data
                            beta_1_conc, beta_2_conc, reg_conc = hybrid_model.settings(X_train_conc, y_train_conc)
                            if beta_2_conc == 0.0:
                                alpha = np.nan
                            else:
                                # if save_model:
                                #    pickle.dumps(reg_conc)
                                alpha = reg_conc.alpha
                            data_conc.update(
                                {f'conc_level_up_to_{test_idx + 1}_size_{len(y_test)}':
                                    {
                                        'conc_test_mut_level': test_idx + 1,
                                        'n_y_train': len(y_train_conc),
                                        'n_y_test': len(y_test),
                                        # 'rnd_state': random_state,
                                        'spearman_rho': hybrid_model.spearmanr(
                                            y_test,
                                            hybrid_model.predict(X_test, reg_conc, beta_1_conc, beta_2_conc)
                                        ),
                                        'beta_1': beta_1_conc,
                                        'beta_2': beta_2_conc,
                                        'alpha': alpha
                                    }
                                }
                            )
                        else:  # ML updating pureML regression model params with newly inputted concatenated train data
                            regressor.fit(X_train_conc, y_train_conc)
                            data.update(
                                {f'single_level_{test_idx + 1}_size_{len(y_test)}':
                                    {
                                        'test_mut_level': test_idx + 1,
                                        'n_y_train': len(y_train),
                                        'n_y_test': len(y_test),
                                        'spearman_rho': stats.spearmanr(
                                            y_test,  # Call predict on the BaseSearchCV estimator
                                            regressor.predict(X_test)  # with the best found parameters
                                        )[0]
                                    }
                                }
                            )

    return data, data_conc
