#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

from Modules_Regression import Predict

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
# ignoring warnings of PLS regression using n_components
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

def mutate_sequence(seq, m, Model, prev_mut_loc, AAs, Sub_LS, iteration, counter, usecsv, csvaa):
    """
    produces a mutant sequence (integer representation), given an initial sequence
    and the number of mutations to introduce ("m") for in silico directed evolution
    """
    try:
        os.mkdir('EvoTraj')
    except FileExistsError:
        pass
    myfile = 'EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_' + str(iteration+1) + '.fasta'
    var_seq_list = []
    with open(myfile, 'w') as mf:
        for i in range(m):  # iterate through number of mutations to add
            rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)  # find random position to mutate
            while (rand_loc <= 0) or (rand_loc >= len(seq)):
                rand_loc = random.randint(prev_mut_loc - 8, prev_mut_loc + 8)

            if usecsv is True:  # Only perform directed evolution on positional csv variant data
                pos_list = []
                aa_list = []
                for aa_positions in Sub_LS:
                    for pos in aa_positions:
                        pos_int = int(pos[1:-1])
                        if pos_int not in pos_list:
                            pos_list.append(pos_int)
                        if csvaa is True:
                            new_aa = str(pos[-1:])
                            if new_aa not in aa_list:
                                aa_list.append(new_aa)
                            AAs = aa_list
                # Select closest Position to single AA positions
                absolute_difference_function = lambda list_value: abs(list_value - rand_loc)
                try:
                    closest_loc = min(pos_list, key=absolute_difference_function)
                except ValueError:
                    raise ValueError("No positions for recombination found. Likely no single "
                                     "substitutional variants found in provided .csv file.")
                rand_loc = closest_loc - 1   # - 1 as Position 17 is 16 when starting with 0 index
            rand_aa = random.choice(AAs)  # find random amino acid to mutate to
            sequence = seq
            seq_ = list(sequence)
            seq_[rand_loc] = rand_aa  # update sequence to have new amino acid at randomly chosen position
            seq_ = ''.join(seq_)
            var = str(rand_loc+1)+str(rand_aa)
            var_seq_list.append([var, seq_])

            print('> {}{}'.format(rand_loc+1, rand_aa), file=mf)
            print(''.join(seq_), file=mf)

    return dict(var_seq_list)   # chose dict as one can easily change to more sequences to predict per iteration


def restructure_dict(prediction_dict):
    """
    Exchange key and value of a dictionary
    """
    restructured = []
    for pred in prediction_dict:
        restruct = []
        for p in pred:
            restruct.insert(0, p)
        restructured.append(restruct)
    structured_dict = dict(restructured)
    return structured_dict


def write_MCMC_predictions(Model, iter, predictions, counter):
    """
    write predictions to EvoTraj folder to .fasta files for each iteration of evolution
    """
    with open('EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_' + str(iter+1)
              + '.fasta', 'r') as f_in:
        with open('EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_' + str(iter+1)
                  + '_prediction.fasta', 'w') as f_out:
            for line in f_in:
                f_out.write(line)
                if '>' in line:
                    key = line[2:].strip()
                    f_out.writelines('; ' + str(predictions.get(key)) + '\n')
    return ()


def in_silico_de(s_WT, num_iterations, Model, amino_acids, T, Path, Sub_LS, counter,
                 noFFT=False, negative=False, usecsv=False, csvaa=False, print_matrix=False):
    """
    Perform directed evolution by randomly selecting a sequence position for substitution and randomly choose the
    amino acid to substitute to. New sequence gets accepted if meeting the Metropolis criterion and will be
    taken for new substitution iteration.
    Metropolis-Hastings-driven directed evolution, similar to Biswas et al.:
    Low-N protein engineering with data-efficient deep learning,
    see https://github.com/ivanjayapurna/low-n-protein-engineering/tree/master/directed-evo
    """
    v_traj = []  # initialize an array to keep records of the variant names for this trajectory
    y_traj = []  # initialize an array to keep records of the fitness scores for this trajectory
    s_traj = []  # initialize an array to keep records of the protein sequences for this trajectory

    # iterate through the trial mutation steps for the directed evolution trajectory
    for i in range(num_iterations):  # num_iterations

        if i == 0:  # get first v, y, s
            # randomly choose the location of the first mutation in the trajectory
            mut_loc_seed = random.randint(0, len(s_WT))
            # m = 1 instead of (np.random.poisson(2) + 1)
            var_seq_dict = mutate_sequence(s_WT, 1, Model, mut_loc_seed, amino_acids, Sub_LS, 0, counter, usecsv, csvaa)

            predictions = Predict(Path, 'EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_'
                                  + str(i+1) + '.fasta', Model, None, noFFT, print_matrix)
            predictions = restructure_dict(predictions)

            write_MCMC_predictions(Model, i, predictions, counter)

            ys, variants = [], []
            for var in predictions:
                variants.append(var)
                ys.append(predictions.get(var))

            y, var = ys[0], variants[0]  # only one entry anyway
            new_mut_loc = int(var[:-1]) - 1
            sequence = var_seq_dict.get(var)

            v_traj.append(var)
            y_traj.append(y)
            s_traj.append(sequence)

        else:  # based on first v, y, s go deeper in mutations --> new_v, new_y, new_s if accepted
            # only chose 1 mutation to introduce and not:
            # mu = np.random.uniform(1, 2.5) --> Number of Mutations = m = np.random.poisson(mu - 1) + 1
            new_var_seq_dict = mutate_sequence(sequence, 1, Model, new_mut_loc, amino_acids,
                                               Sub_LS, i, counter, usecsv, csvaa)

            predictions = Predict(Path, 'EvoTraj/' + str(Model) + '_EvoTraj_' + str(counter+1) + '_DEiter_'
                                  + str(i+1) + '.fasta', Model, None, noFFT, print_matrix)
            predictions = restructure_dict(predictions)

            write_MCMC_predictions(Model, i, predictions, counter)

            new_ys, new_variants = [], []
            for var in predictions:
                new_variants.append(var)
                new_ys.append(predictions.get(var))

            new_y, new_y_var = new_ys[0], new_variants[0]
            new_mut_loc = int(new_y_var[:-1]) - 1
            new_sequence = new_var_seq_dict.get(new_y_var)

            # probability function for trial sequence
            # The lower the fitness (y) of the new variant, the higher are the chances to get excluded
            with warnings.catch_warnings():  # catching Overflow warning
                warnings.simplefilter("ignore")
                try:
                    boltz = np.exp(((new_y - y) / T), dtype=np.longfloat)
                    if negative is True:
                        boltz = np.exp(((-new_y - -y) / T), dtype=np.longfloat)
                except OverflowError:
                        boltz = 1
            p = min(1, boltz)
            rand_var = random.random()  # random float between 0 and 1
            if rand_var < p:  # Metropolis-Hastings update selection criterion
                # print('Updated sequence as: Rand ({}) < Boltz ({})'.format(str(rand_var), str(boltz)))
                # print(str(new_mut_loc + 1) + " " + sequence[new_mut_loc] + "->" + new_sequence[new_mut_loc])
                var, y, sequence = new_y_var, new_y, new_sequence  # if criteria is met, update sequence and
                                                                   # corresponding fitness
                v_traj.append(var)  # update the variant naming trajectory records for this iteration of mutagenesis
                y_traj.append(y)  # update the fitness trajectory records for this iteration of mutagenesis
                s_traj.append(sequence)  # update the sequence trajectory records for this iteration of mutagenesis

    return v_traj, s_traj, y_traj


def run_DE_trajectories(s_wt, Model, y_WT, num_iterations, num_trajectories, DE_record_folder, amino_acids, T, Path,
                        Sub_LS, noFFT=False, negative=False, save=False, usecsv=False, csvaa=False, print_matrix=False):
    """
    Runs the directed evolution by adressing the in_silico_de function and plots the evolution trajectories.
    """
    v_records = []  # initialize list of sequence variant names
    s_records = []  # initialize list of sequence records
    y_records = []  # initialize list of fitness score records


    for i in range(num_trajectories):  #iterate through however many mutation trajectories we want to sample
        # call the directed evolution function, outputting the trajectory sequence and fitness score records
        v_traj, s_traj, y_traj = in_silico_de(s_wt, num_iterations, Model, amino_acids, T, Path, Sub_LS, i,
                                              noFFT, negative, usecsv, csvaa, print_matrix)

        v_records.append(v_traj)    # update the variant naming trajectory records for this full mutagenesis trajectory
        s_records.append(s_traj)  # update the sequence trajectory records for this full mutagenesis trajectory
        y_records.append(y_traj)  # update the fitness trajectory records for this full mutagenesis trajectory

        if save==True:
            try:
                os.mkdir(DE_record_folder)
            except FileExistsError:
                pass
            # save sequence records for trajectory i
            np.savetxt(DE_record_folder + "/" + str(Model) + "_trajectory" + str(i+1)
                       + "_seqs.txt", np.array(s_traj), fmt="%s")
            # save fitness records for trajectory i
            np.savetxt(DE_record_folder + "/" + str(Model) + "_trajectory" + str(i+1)
                       + "_fitness.txt", np.array(y_traj))

    # numpy warning filter needed for arraying ragged nested sequences
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    v_records = np.array(v_records)
    s_records = np.array(s_records)
    y_records = np.array(y_records)

    fig, ax = plt.subplots()  # figsize=(10, 6)
    ax.locator_params(integer=True)
    f_len_max = 0
    for j, f in enumerate(y_records):
        if y_WT is not None:
            f_len = len(f)
            if f_len > f_len_max:
                f_len_max = f_len
            ax.plot(np.arange(1, len(f)+2, 1), np.insert(f, 0, y_WT),
                    '-o', alpha=0.7, markeredgecolor='black', label='EvoTraj' + str(j+1))
        else:
            ax.plot(np.arange(1, len(f)+1, 1), f,
                    '-o', alpha=0.7, markeredgecolor='black', label='EvoTraj' + str(j+1))

    label_x_y_name = []
    for k, l in enumerate(v_records):
        for kk, ll in enumerate(l):  # kk = 1, 2, 3, ...  (=x); ll = variant name; y_records[k][kk] = fitness (=y)
            if y_WT is not None:     # kk+2 as enumerate starts with 0 and WT is 1 --> start labeling with 2
                label_x_y_name.append(ax.text(kk+2, y_records[k][kk], ll))
            else:
                label_x_y_name.append(ax.text(kk+1, y_records[k][kk], ll))

    from adjustText import adjust_text
    adjust_text(label_x_y_name, only_move={'points': 'y', 'text': 'y'}, force_points=0.5)
    leg = ax.legend()
    if y_WT is not None:
        wt_tick = (['', 'WT'] + ((np.arange(1, f_len_max+1, 1)).tolist()))
        with warnings.catch_warnings():  # catching matplotlib 3.3.1 UserWarning
            warnings.simplefilter("ignore")
            ax.set_xticklabels(wt_tick)

    plt.ylabel('Predicted Fitness')
    plt.xlabel('Mutation Trial Steps')
    plt.savefig(str(Model) + '_DE_trajectories.png')

    return s_records, y_records