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

import os
import sys
import numpy as np
import pandas as pd


amino_acids = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]


def read_models(number):
    """
    reads the models found in the file Model_Results.txt.
    If no model was trained, the .txt file does not exist.
    """
    try:
        ls = ""
        with open('Model_Results.txt', 'r') as file:
            for i, lines in enumerate(file):
                if i == 0:
                    if lines[:6] == 'No FFT':
                        number += 2
                if i <= number + 1:
                    ls += lines
        return ls
    except FileNotFoundError:
        return "No Model_Results.txt found."


def full_path(filename):
    """
    returns the path of an index inside the folder /AAindex/,
    e.g. path/to/AAindex/FAUJ880109.txt.
    """
    modules_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(modules_path, '../AAindex/' + filename)


def path_aaindex_dir():
    """
    returns the absolute path to the /AAindex folder,
    e.g. c/users/name/path/to/AAindex/.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '../AAindex')


def get_sequences_from_file(fasta, mult_path=None):
    """
    "Get_Sequences" reads (learning and test).fasta format
    files and extracts the name, the target value and the
    sequence of the peptide. See example directory for required
    fasta file format. Make sure every marker (> and ;) is
    seperated by an space ' ' from the value respectively name.
    """
    if mult_path is not None:
        os.chdir(mult_path)

    sequences = []
    values = []
    names_of_mutations = []

    with open(fasta, 'r') as f:
        for line in f:
            if '>' in line:
                words = line.split()
                names_of_mutations.append(words[1])
                # words[1] is appended so make sure there is a space in between > and the name!

            elif '#' in line:
                pass  # are Comments

            elif ';' in line:
                words = line.split()
                values.append(float(words[1]))
                # words[1] is appended so make sure there is a space in between ; and the value!

            else:
                try:
                    words = line.split()
                    sequences.append(words[0])
                except IndexError:
                    raise IndexError("Learning or Validation sets (.fasta) likely "
                                     "have emtpy lines (e.g. at end of file)")

    # Check consistency
    if len(values) != 0:
        if len(sequences) != len(values):
            print('Error: Number of sequences does '
                  'not fit with number of target values!')
            print('Number of sequences: {}, Number of target values: {}.'.format(
                str(len(sequences)), str(len(values))))
            sys.exit()

    return sequences, names_of_mutations, values


def remove_nan_encoded_positions(xs: list, ys: list = None):
    """
    Removes encoded sequence (x) of sequence list xs when NaNs occur in x.
    Also removes the corresponding fitness value y (f(x) --> y) at position i.
    ys can als be any type of list, e.g. variants or sequences.
    """
    if ys is None:
        ys = np.zeros(len(xs))
    drop = []
    for i, x in enumerate(xs):
        if None in x:
            drop.append(i)
    drop = sorted(drop, reverse=True)
    for idx in drop:
        del xs[idx]
        del ys[idx]
    return xs, ys


def get_basename(filename: str) -> str:
    """
    Description
    -----------
    Extracts and returns the basename of the filename.

    Parameters
    ----------
    filename: str

    Returns
    -------
    str
    """
    return os.path.basename(filename).split('.')[0]


def read_csv(
        file_name: str,
        fitness_key: str = None
) -> tuple[list, list, list]:
    """
    Description
    -----------
    Reads input CSV file and return variants names and
    associated fitness values.

    Parameters
    ----------
    file_name: str
        Name of CSV file to read.
    fitness_key: str
        Name of column containing the fitness values.
        If None, column 1 (0-indexed) will be taken.

    Returns
    -------
    variants: np.ndarray
        Array of variant names
    fitnesses:
        Array of fitness values
    """
    df = pd.read_csv(file_name, sep=';', comment='#')
    if df.shape[1] == 1:
        df = pd.read_csv(file_name, sep=',', comment='#')
    if fitness_key is not None:
        fitnesses = df[fitness_key].to_numpy(dtype=float)
    else:
        fitnesses = list(df.iloc[:, 1].to_numpy(dtype=float))
    variants = list(df.iloc[:, 0].to_numpy(dtype=str))
    features = list(df.iloc[:, 2:].to_numpy(dtype=float))

    return variants, fitnesses, features


def generate_dataframe_and_save_csv(
        variants,
        sequence_encodings,
        fitnesses,
        csv_file: str,
        save_df_as_csv: bool = True
) -> pd.DataFrame:
    """
    Description
    -----------
    Creates a pandas.DataFrame from the input data (numpy array including
    variant names, fitnesses, and encoded sequences).
    Writes pandas.DataFrame to a specified CSV file follwing the scheme:
    variants; fitness values; encoded sequences

    Parameters
    ----------
    variants: list
        Variant names.
    fitnesses: list
        Sequence-associated fitness value.
    Encoded sequence: list
        Sequence encodings (feature matrix) of sequences.
    csv_file : str
        Name of the csv file containing variant names and associated fitness values.
    save_df_as_csv : bool
        Writing DataFrame (Substitution;Fitness;Encoding_Features) to CSV (False/True).

    Returns
    -------
    df_dca: pandas.DataFrame
        Dataframe with variant names, fitness values, and features (encoded sequences).
        If save_df_as_csv is True also writes DF to CSV.
    """
    X = np.stack(sequence_encodings)                                  # construction
    feature_dict = {}            # Collecting features for each MSA position i
    for i in range(X.shape[1]):  # (encoding at pos. i) in a dict
        feature_dict[f'X{i + 1:d}'] = X[:, i]

    df_dca = pd.DataFrame()
    df_dca.insert(0, 'variant', variants)
    df_dca.insert(1, 'y', fitnesses)
    df_dca = pd.concat([df_dca, pd.DataFrame(feature_dict)], axis=1)

    if save_df_as_csv:
        filename = f'{get_basename(csv_file)}_encoded.csv'
        df_dca.to_csv(filename, sep=';', index=False)

    return df_dca


def process_df_encoding(df_encoding) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the array of names, encoded sequences, and fitness values
    of the variants from the dataframe 'self.df_encoding'.
    It is mandatory that 'df_encoding' contains the names of the
    variants in the first column, the associated fitness value in the
    second column, and the encoded sequence starting from the third
    column.
    Returns
    -------
    Tuple of variant names, encoded sequences, and fitness values.
    """
    return (
        df_encoding.iloc[:, 0].to_numpy(),
        df_encoding.iloc[:, 2:].to_numpy(),
        df_encoding.iloc[:, 1].to_numpy()
    )


def count_mutation_levels_and_get_dfs(df_encoding) -> tuple:
    """
    """
    single_variants_index, all_higher_variants_index = [], []
    double_i, triple_i, quadruple_i, quintuple_i, sextuple_i, \
    septuple_i, octuple_i, nonuple_i, higher_nine_i = [], [], [], [], [], [], [], [], []
    for i, row in enumerate(df_encoding.iloc[:, 0]):  # iterate over variant column
        if '/' in row:  # TypeError: argument of type 'float' is not iterable if empty columns are (at end of) CSV
            all_higher_variants_index.append(i)
            if row.count('/') == 1:
                double_i.append(i)
            elif row.count('/') == 2:
                triple_i.append(i)
            elif row.count('/') == 3:
                quadruple_i.append(i)
            elif row.count('/') == 4:
                quintuple_i.append(i)
            elif row.count('/') == 5:
                sextuple_i.append(i)
            elif row.count('/') == 6:
                septuple_i.append(i)
            elif row.count('/') == 7:
                octuple_i.append(i)
            elif row.count('/') == 8:
                nonuple_i.append(i)
            elif row.count('/') >= 9:
                higher_nine_i.append(i)
        else:
            single_variants_index.append(i)
    print(f'No. Singles: {len(single_variants_index)}\nNo. All higher: {len(all_higher_variants_index)}\n'
          f'2: {len(double_i)}\n3: {len(triple_i)}\n4: {len(quadruple_i)}\n'
          f'5: {len(quintuple_i)}\n6: {len(sextuple_i)}\n7: {len(septuple_i)}\n'
          f'8: {len(octuple_i)}\n9: {len(nonuple_i)}\n>=10: {len(higher_nine_i)}')
    return (
        df_encoding.iloc[single_variants_index, :],
        df_encoding.iloc[double_i, :],
        df_encoding.iloc[triple_i, :],
        df_encoding.iloc[quadruple_i, :],
        df_encoding.iloc[quintuple_i, :],
        df_encoding.iloc[sextuple_i, :],
        df_encoding.iloc[septuple_i, :],
        df_encoding.iloc[octuple_i, :],
        df_encoding.iloc[nonuple_i, :],
        df_encoding.iloc[higher_nine_i, :],
        df_encoding.iloc[all_higher_variants_index, :],
    )


