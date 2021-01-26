from .NK_landscape import makeNK, hamming, collapse_single
import pandas as pd
import numpy as np
import os

def gen_distance_subsets(ruggedness,seq_len=5,library="ACDEFGHIKL",seed=None):
    """
    Takes a ruggedness, sequence length, and library and produces an NK landscape then separates it
    into distances from a seed sequence.

    ruggedness [int | 0-(seq_len-1)]  : Determines the ruggedness of the landscape
    seq_len : length of all of the sequences
    library : list of possible characters in strings
    seed    : the seed sequence for which distances will be calculated

    returns ->  {distance : [(sequence,fitness)]}
    """

    land_K2, seq, _ = makeNK(seq_len,ruggedness,library)

    if not seed:
        seed = np.array([x for x in "".join([library[0] for x in range(seq_len)])])

    subsets = {x : [] for x in range(seq_len+1)}
    for seq in land_K2:
        subsets[hamming(seq[0],seed)].append(seq)

    return subsets

def dataset_generation(directory="../Data",seq_len=5):
    """
    Generates five instances of each possible ruggedness value for the NK landscape

    seq_len
    """

    if not os.path.exists(directory):
        os.mkdir(directory)

    datasets = {x : [] for x in range(seq_len)}

    for ruggedness in range(0,seq_len):
        for instance in range(5):
            print("Generating data for K={} V={}".format(ruggedness,instance))

            subsets = gen_distance_subsets(ruggedness,seq_len)

            hold = []

            for i in subsets.values():
                for j in i:
                    hold.append([collapse_single(j[0]),j[1]])

            saved = np.array(hold)
            df = pd.DataFrame({"Sequence" : saved[:,0], "Fitness" : saved[:,1]})
            df.to_csv("{0}/K{1}/V{2}.csv".format(directory,ruggedness,instance))

    print ("All data generated. Data is stored in: {}".format(directory))

if __name__ == "__main__":
    answer = input("Rerunning this script as a standalone will generate all data again in the default directory, is this what is desired? y/[n]")
    if answer.lower() == "y" or answer.lower() == "yes":
        dataset_generation()
