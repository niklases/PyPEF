#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 05 October 2020
# @author: Niklas Siedhoff, Alexander-Maurice Illig
# <n.siedhoff@biotec.rwth-aachen.de>, <a.illig@biotec.rwth-aachen.de>
# PyPEF - Pythonic Protein Engineering Framework
# Released under Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)
# For more information about the license see https://creativecommons.org/licenses/by-nc/4.0/legalcode

import os
import numpy as np
from tqdm import tqdm


def Make_Fasta_PS(filename, WT, Sub):
    """
    Creates prediction sets (.fasta style files)
    """
    myfile = open(filename, 'w')
    Count = 0
    for i in Sub:
        Temp = list(WT)
        name = ''
        b = 0
        for j in i:
            Position_Index = int(str(j)[1:-1]) - 1
            New_Amino_Acid = str(j)[-1]
            Temp[Position_Index] = New_Amino_Acid
            if b == 0:
                name += j
            else:
                name += '/' + j
            b += 1
        print('>', name, file=myfile)
        print(''.join(Temp), file=myfile)
        Count += 1
    myfile.close()


def Make_Combinations_Double(Arr):
    """
    Make double recombination variants
    """
    Doubles = []
    for i in tqdm(range(len(Arr))):
        for j in range(len(Arr)):
            if j > i:
                if (Arr[i][0])[1:-1] != (Arr[j][0])[1:-1]:
                    Doubles.append([Arr[i][0], Arr[j][0]])
                    if len(Doubles) >= 8E04:
                        yield Doubles
                        Doubles = []
    yield Doubles


def Make_Combinations_Triple(Arr):
    """
    Make triple recombination variants
    """
    length = len(Arr)
    Triples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                if k > j > i:
                    if (Arr[i][0])[1:-1] != (Arr[j][0])[1:-1] != (Arr[k][0])[1:-1]:
                        Triples.append([Arr[i][0], Arr[j][0], Arr[k][0]])
                        if len(Triples) >= 8E04:
                            yield Triples
                            Triples = []
    yield Triples


def Make_Combinations_Quadruple(Arr):
    """
    Make quadruple recombination variants
    """
    length = len(Arr)
    Quadruples = []
    for i in tqdm(range(length)):
        for j in range(length):
            for k in range(length):
                for l in range(length):
                    if l > k > j > i:
                        if (Arr[i][0])[1:-1] != (Arr[j][0])[1:-1] != (Arr[k][0])[1:-1] != (Arr[l][0])[1:-1]:
                            Quadruples.append([Arr[i][0], Arr[j][0], Arr[k][0], Arr[l][0]])
                            if len(Quadruples) >= 8E04:
                                yield Quadruples
                                Quadruples = []
    yield Quadruples


def Make_Directory_And_Enter(Directory):
    """
    Makes directory for recombined or diverse prediction sets
    """
    Previous_Working_Directory = os.getcwd()
    try:
        if not os.path.exists(os.path.dirname(Directory)):
            os.mkdir(Directory)
    except OSError:
        pass
    os.chdir(Directory)

    return (Previous_Working_Directory)


def create_split_files(array, single_variants, WT_Sequence, name, no):
    """
    Creates split files from given variants for yielded recombined or diverse variants
    """
    if len(array) > 0:
        Number_Of_Split_Files = len(array) / (len(single_variants) * 20 ** 3)
        Number_Of_Split_Files = round(Number_Of_Split_Files)
        if Number_Of_Split_Files == 0:
            Number_Of_Split_Files += 1
        Split = np.array_split(array, Number_Of_Split_Files)
        pwd = Make_Directory_And_Enter(name + '_Split')
        for i in Split:
            name_ = name + '_Split' + str(no) + '.fasta'
            Make_Fasta_PS(name_, WT_Sequence, i)

        os.chdir(pwd)


def Make_Combinations_Double_All_Diverse(Arr, Aminoacids):
    """
    Make double substituted naturally diverse variants
    """
    Doubles = []
    for i in tqdm(range(len(Arr))):
        for j in range(i + 1, len(Arr)):
            for k in Aminoacids:
                for l in Aminoacids:
                    if ((Arr[i][0])[1:-1]) != ((Arr[j][0])[1:-1]) and\
                            ((Arr[i][0])[:-1] + k)[0] != ((Arr[i][0])[:-1] + k)[-1] and\
                            ((Arr[j][0])[:-1] + l)[0] != ((Arr[j][0])[:-1] + l)[-1]:
                            Doubles.append(tuple([(Arr[i][0])[:-1] + k, (Arr[j][0])[:-1] + l]))  # tuple needed for
                            if len(Doubles) >= 8E04:                                             # list(dict()):
                                Doubles = list(dict.fromkeys(Doubles))  # transfer to dict removes duplicated list entries
                                yield Doubles
                                Doubles = []
    Doubles = list(dict.fromkeys(Doubles))
    yield Doubles


def Make_Combinations_Triple_All_Diverse(Arr, Aminoacids):
    """
    Make triple substituted naturally diverse variants
    """
    Triples = []
    for i in tqdm(range(len(Arr))):
        for j in range(i + 1, len(Arr)):
            for k in range(j + 1, len(Arr)):
                for l in Aminoacids:
                    for m in Aminoacids:
                        for n in Aminoacids:
                            if ((Arr[i][0])[1:-1]) != ((Arr[j][0])[1:-1]) != ((Arr[k][0])[1:-1]) and\
                                    ((Arr[i][0])[:-1] + l)[0] != ((Arr[i][0])[:-1] + l)[-1] and\
                                    ((Arr[j][0])[:-1] + m)[0] != ((Arr[j][0])[:-1] + m)[-1] and\
                                    ((Arr[k][0])[:-1] + n)[0] != ((Arr[k][0])[:-1] + n)[-1]:
                                    Triples.append(tuple([(Arr[i][0])[:-1] + l, (Arr[j][0])[:-1] + m, (Arr[k][0])[:-1] + n]))
                                    if len(Triples) >= 8E04:
                                        Triples = list(dict.fromkeys(Triples))  # transfer to dict and back to list
                                        yield Triples
                                        Triples = []
    Triples = list(dict.fromkeys(Triples))
    yield Triples



def Make_Combinations_Quadruple_All_Diverse(Arr, Aminoacids):
    """
    Make quadruple substituted naturally diverse variants
    """
    Quadruples = []
    for i in tqdm(range(len(Arr))):
        for j in range(i + 1, len(Arr)):
            for k in range(j + 1, len(Arr)):
                for l in range(k + 1, len(Arr)):
                    for m in Aminoacids:
                        for n in Aminoacids:
                            for o in Aminoacids:
                                for p in Aminoacids:
                                    if ((Arr[i][0])[1:-1]) != ((Arr[j][0])[1:-1]) != ((Arr[k][0])[1:-1]) != ((Arr[l][0])[1:-1]) and\
                                            ((Arr[i][0])[:-1] + m)[0] != ((Arr[i][0])[:-1] + m)[-1] and\
                                            ((Arr[j][0])[:-1] + n)[0] != ((Arr[j][0])[:-1] + n)[-1] and\
                                            ((Arr[k][0])[:-1] + o)[0] != ((Arr[k][0])[:-1] + o)[-1] and\
                                            ((Arr[l][0])[:-1] + p)[0] != ((Arr[l][0])[:-1] + p)[-1]:
                                                Quadruples.append(tuple([(Arr[i][0])[:-1] + m, (Arr[j][0])[:-1] + n,
                                                                         (Arr[k][0])[:-1] + o, (Arr[l][0])[:-1] + p]))
                                                if len(Quadruples) >= 8E04:
                                                    Quadruples = list(dict.fromkeys(Quadruples))  # transfer to dict and back to list
                                                    yield Quadruples
                                                    Quadruples = []
    Quadruples = list(dict.fromkeys(Quadruples))
    yield Quadruples
