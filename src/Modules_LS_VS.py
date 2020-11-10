#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

import numpy as np
import random
import pandas as pd
import re
import glob


def get_wt_sequence(sequence_file):
    """
    Gets wild-type sequence from defined input file (can be pure sequence or fasta style)
    """
    # In wild-type sequence .fa file one has to give the sequence of the studied peptide/protein (wild-type)
    # If no .csv is defined is input this tries to find and select any .fa file
    # (risk: could lead to errors if several .fa files exist in folder)
    if sequence_file is None or sequence_file not in glob.glob(sequence_file):
        try:
            types = ('wt_sequence.*', 'wild_type_sequence.*', '*.fa')  # the tuple of file types
            files_grabbed = []
            for files in types:
                files_grabbed.extend(glob.glob(files))
            for i, file in enumerate(files_grabbed):
                if i == 0:
                    sequence_file = file
            if len(files_grabbed) == 0:
                raise FileNotFoundError("Found no input wild-type fasta sequence file (.fa) in current directory!")
            print('Did not find (specified) WT-sequence file! Used wild-type sequence file instead:'
                  ' {}.'.format(str(sequence_file)))
        except NameError:
            raise NameError("Found no input wild-type fasta sequence file (.fa) in current directory!")
    Wild_Type_Sequence = ""
    with open(sequence_file, 'r') as sf:
        for lines in sf.readlines():
            if lines.startswith(">"):
                continue
            lines = ''.join(lines.split())
            Wild_Type_Sequence += lines
    return Wild_Type_Sequence


def csv_input(csv_file):
    """
    Gets input data from defined .csv file (that contains variant names and fitness labels)
    """
    if csv_file is None or csv_file not in glob.glob(csv_file):
        for i, file in enumerate(glob.glob('*.csv')):
            if file.endswith('.csv'):
                if i == 0:
                    csv_file = file
                    print('Did not find (specified) csv file! Used csv input file instead: {}.'.format(str(csv_file)))
        if len(glob.glob('*.csv')) == 0:
            raise FileNotFoundError('Found no input .csv file in current directory.')
    return csv_file


def drop_rows(csv_file, amino_acids, threshold_drop):
    """
    Drops rows from .csv data if below defined fitness threshold or if
    amino acid/variant name is unknown or if fitness label is not a digit.
    """
    separator = ';'
    try:
        df_raw = pd.read_csv(csv_file, sep=separator, usecols=[0, 1])
    except ValueError:
        separator = ','
        df_raw = pd.read_csv(csv_file, sep=separator, usecols=[0, 1])

    label = df_raw.iloc[:, 1]
    sequence = df_raw.iloc[:, 0]

    drop_rows = []

    for i, row in enumerate(label):
        try:
            row = float(row)
            if row < threshold_drop:
                drop_rows.append(i)
        except ValueError:
            drop_rows.append(i)

    for i, variant in enumerate(sequence):
        try:
            if '/' in variant:
                m = re.split(r'/', variant)
                for a, splits in enumerate(m):
                    if splits[0].isdigit() and variant[-1] in amino_acids:
                        continue
                    elif splits[0] not in amino_acids or splits[-1] not in amino_acids:
                        if i not in drop_rows:
                            drop_rows.append(i)
                            # print('Does not know this definition of amino acid substitution: Variant:', variant)
            else:
                if variant[0].isdigit() and variant[-1] in amino_acids:
                    continue
                elif variant[0] not in amino_acids or variant[-1] not in amino_acids:
                    drop_rows.append(i)
                    # print('Does not know this definition of amino acid substitution: Variant:', variant)
        except TypeError:
            raise TypeError('You might consider checking the input .csv for empty first two columns,'
                            ' e.g. in the last row.')

    print('No. of dropped rows: {}.'.format(len(drop_rows)), 'Total given variants: {}'.format(len(df_raw)))

    df = df_raw.drop(drop_rows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_variants(df, amino_acids, Wild_Type_Sequence):
    """
    Gets variants and divides and counts the variant data for single substituted and higher substituted variants.
    Raises NameError if variant naming is not matching the given wild-type sequence, e.g. if variant A17C would define
    a substitution at residue Ala-17 to Cys but the wild-type sequence has no Ala at position 17.
    """
    X = df.iloc[:, 0]
    Y = df.iloc[:, 1]
    single_variants, higher_variants, index_higher, index_lower, higher_values, single_values = [], [], [], [], [], []
    single, double, triple, quadruple, quintuple, sextuple, \
    septuple, octuple, nonuple, decuple, higher = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i, variant in enumerate(X):
        if '/' in variant:
            count = variant.count('/')
            if count == 1:
                double += 1
            elif count == 2:
                triple += 1
            elif count == 3:
                quadruple += 1
            elif count == 4:
                quintuple += 1
            elif count == 5:
                sextuple += 1
            elif count == 6:
                septuple += 1
            elif count == 7:
                octuple += 1
            elif count == 8:
                nonuple += 1
            elif count == 9:
                decuple += 1
            else:
                higher += 1
            m = re.split(r'/', variant)
            for a, splits in enumerate(m):
                if splits[0].isdigit() or splits[0] in amino_acids and splits[-1] in amino_acids:
                    new = int(re.findall(r'\d+', splits)[0])
                    if splits[0] in amino_acids:
                        if splits[0] != Wild_Type_Sequence[new - 1]:
                            raise NameError('Position of amino acids in given sequence does not match the given '
                                            'positions in the input data! E.g. see position {} and position {} being {} '
                                            'in the given sequence'.format(variant, new, Wild_Type_Sequence[new - 1]))
                    higher_var = Wild_Type_Sequence[new - 1] + str(new) + str(splits[-1])
                    m[a] = higher_var
                    if a == len(m) - 1:
                        higher_variants.append(m)
                        if i not in index_higher:
                            index_higher.append(i)
        else:
            single += 1
            if variant[0].isdigit() or variant[0] in amino_acids and variant[-1] in amino_acids:
                num = int(re.findall(r'\d+', variant)[0])
                if variant[0] in amino_acids:
                    if variant[0] != Wild_Type_Sequence[num - 1]:
                        raise NameError('Position of amino acids in given sequence does not match the given '
                                        'positions in the input data! E.g. see position {} and position {} being {} '
                                        'in the given sequence.'.format(variant, num, Wild_Type_Sequence[num - 1]))
                full_variant = Wild_Type_Sequence[num - 1] + str(num) + str(variant[-1])
                single_variants.append([full_variant])
                if i not in index_lower:
                    index_lower.append(i)
    print('Single: {}.'.format(single), 'Double: {}.'.format(double), 'Triple: {}.'.format(triple),
          'Quadruple: {}.'.format(quadruple), 'Quintuple: {}.'.format(quintuple), 'Sextuple: {}.'.format(sextuple),
          'Septuple: {}.'.format(septuple), 'Octuple: {}.'.format(octuple), 'Nonuple: {}.'.format(nonuple),
          'Decuple: {}.'.format(decuple), 'Higher: {}.'.format(higher))
    for vals in Y[index_higher]:
        higher_values.append(vals)
    for vals in Y[index_lower]:
        single_values.append(vals)

    single_variants, single_values = tuple(single_variants), tuple(single_values)
    higher_variants, higher_values = tuple(higher_variants), tuple(higher_values)

    return single_variants, single_values, higher_variants, higher_values


def make_sub_LS_VS(single_variants, single_values, higher_variants, higher_values, directed_evolution=False):
    """
    Creates learning and validation sets, fills learning set with single substituted variants and splits
    rest (higher substituted) for learning and validation sets: 3/4 to LS and 1/4 to VS
    """
    print('No. of single subst. variants: {}.'.format(len(single_variants)),
          'No. of values: {}'.format(len(single_values)))
    print('No. of higher subst. variants: {}.'.format(len(higher_variants)),
          'No. of values: {}'.format(len(higher_values)))

    if len(single_values) != len(single_variants):
        print('Error due to different lengths for given variants and label!'
              ' No. of single subst. variants: {}.'.format(len(single_variants)),
              ' Number of given values: {}.'.format(len(single_values)))

    if len(higher_values) != len(higher_variants):
        print('Error due to different lengths for given variants and label! No. of higher subst. variants: {}.'
              .format(len(higher_variants)), ' Number of given values: {}.'.format(len(higher_values)))

    # 1. CREATION OF LS AND VS SPLIT FOR SINGLE FOR LS AND HIGHER VARIANTS FOR VS
    Sub_LS = list(single_variants)  # Substitutions of LS
    Val_LS = list(single_values)    # Values of LS

    Sub_VS = []                     # Substitutions of VS
    Val_VS = []                     # Values of VS

    if directed_evolution is False:
        for i in range(len(higher_variants)):
            if len(higher_variants) < 6:  # if less than 6 higher variants all higher variants are appended to VS
                Sub_VS.append(higher_variants[i])
                Val_VS.append(higher_values[i])
            elif (i % 3) == 0 and i is not 0:  # 1/4 of higher variants to VS, 3/4 to LS - change here for LS/VS ratio change
                Sub_VS.append(higher_variants[i])
                Val_VS.append(higher_values[i])
            else:                       # 3/4 to LS
                Sub_LS.append(higher_variants[i])
                Val_LS.append(higher_values[i])

    return Sub_LS, Val_LS, Sub_VS, Val_VS


def make_sub_LS_VS_randomly(single_variants, single_values, higher_variants, higher_values):
    """
    Creation of learning set and validation set by randomly splitting sets
    """
    length = len(single_variants) + len(higher_variants)
    range_list = np.arange(0, length)

    vs = []
    ls = []
    while len(ls) < length * 4 // 5:
        random_num = random.choice(range_list)
        if random_num not in ls:
            ls.append(random_num)

    for j in range_list:
        if j not in ls:
            vs.append(j)

    Combined = single_variants + higher_variants  # Substitutions
    Combined2 = single_values + higher_values  # Values

    Sub_LS = []
    Val_LS = []
    tot_Sub_LS, tot_Val_LS = [], []
    tot_Sub_VS, tot_Val_VS = [], []

    for i in ls:
        Sub_LS.append(Combined[i])
        Val_LS.append(Combined2[i])

    Sub_VS = []
    Val_VS = []
    for j in vs:
        Sub_VS.append(Combined[j])
        Val_VS.append(Combined2[j])

    for subs in Sub_LS:
        for subs2 in Sub_VS:
            if subs == subs2:
                print('\n<Warning> LS and VS overlap for: {} - You might want to consider checking the provided datasets'
                      ' for multiple entries'.format(subs), end=' ')

    tot_Sub_LS.append(Sub_LS)
    tot_Val_LS.append(Val_LS)
    tot_Sub_VS.append(Sub_VS)
    tot_Val_VS.append(Val_VS)

    return tot_Sub_LS[0], tot_Val_LS[0], tot_Sub_VS[0], tot_Val_VS[0]


def make_fasta_LS_VS(filename, WT, Sub, Val):   # Sub = Substitution, Val = Value
    """
    Creates learning and validation sets (.fasta style files)
    """
    myfile = open(filename, 'w')
    Count = 0
    for i in Sub:
        temp = list(WT)
        name = ''
        b = 0
        for j in i:
            Position_Index = int(str(j)[1:-1]) - 1
            New_Amino_Acid = str(j)[-1]
            temp[Position_Index] = New_Amino_Acid
            if b == 0:
                name += j
            else:
                name += '/' + j
            b += 1
        print('>', name, file=myfile)
        print(';', Val[Count], file=myfile)
        print(''.join(temp), file=myfile)
        Count += 1
    myfile.close()
