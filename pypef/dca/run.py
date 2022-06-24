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

from pypef.utils.variant_data import read_csv
from pypef.dca.encoding import DCAEncoding, get_dca_data_parallel
from pypef.dca.model import performance_ls_ts, predict_ps, get_model_and_save_pkl
from pypef.utils.low_n_mutation_extrapolation import performance_mutation_extrapolation, low_n


def run_pypef_hybrid_modeling(arguments):
    threads = abs(arguments['--threads']) if arguments['--threads'] is not None else 1
    threads = threads + 1 if threads == 0 else threads
    print(f'Using {threads} thread(s) for running...')
    print(f"Note that the hybrid model only optimizes model performances in terms of "
          f"Spearman's correlation of measured versus predicted values. Further, the "
          f"hybrid approach uses only Ridge regression for supervised ML-based hybrid "
          f"model contribution. In hybrid modeling, the ranks of predictions are "
          f"important and not the exact predicted value.")

    if arguments['--ls']:
        performance_ls_ts(
            ls_fasta=arguments['--ls'],
            ts_fasta=arguments['--ts'],
            threads=threads,
            starting_position=arguments['--start'],
            params_file=arguments['--params'],
            separator=arguments['--mutation_sep'],
            label=arguments['--label']
        )

    if arguments['--params'] and arguments['--model']:
        prediction_dict = {}
        prediction_dict.update({
            'drecomb': arguments['--drecomb'],
            'trecomb': arguments['--trecomb'],
            'qarecomb': arguments['--qarecomb'],
            'qirecomb': arguments['--qirecomb'],
            'ddiverse': arguments['--ddiverse'],
            'tdiverse': arguments['--tdiverse'],
            'qdiverse': arguments['--qdiverse']
        })

        predict_ps(
            prediction_dict=prediction_dict,
            params_file=arguments['--params'],
            prediction_set=arguments['--ps'],
            test_set=arguments['--ts'],
            threads=threads,
            starting_position=arguments['--start'],
            separator=arguments['--mutation_sep'],
            regressor_pkl=arguments['--model']
        )

    if arguments['train_and_save']:
        dca_encode = DCAEncoding(
            starting_position=arguments['--start'],
            params_file=arguments['--params'],
            separator=arguments['--mutation_sep']
        )

        variants, fitnesses, _ = read_csv(
            arguments['--input']
        )

        variants, encoded_sequences, fitnesses = get_dca_data_parallel(
            variants=variants,
            fitnesses=fitnesses,
            dca_encode=dca_encode,
            threads=threads,
        )

        get_model_and_save_pkl(
            Xs=encoded_sequences,
            ys=fitnesses,
            dca_encoding_instance=dca_encode,
            train_percent_fitting=arguments['--train_size'],
            test_percent=arguments['--test_size']
        )  # CONSTRUCTION

    if arguments['low_n'] or arguments['extrapolation']:
        if arguments['low_n']:
            low_n(
                encoded_csv=arguments['--input'],
                hybrid_modeling=arguments['hybrid']
            )

        elif arguments['extrapolation']:
            performance_mutation_extrapolation(
                encoded_csv=arguments['--input'],
                cv_regressor=arguments['regressor'],
                conc=arguments['--conc']
            )
