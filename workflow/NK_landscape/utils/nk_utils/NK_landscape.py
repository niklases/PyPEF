import numpy as np
import itertools

def collapse_single(protein):
    """
    Takes any iterable form of a single amino acid character sequence and returns a string representing that sequence.
    """
    return "".join([str(i) for i in protein])

def hamming(str1, str2):
    """Calculates the Hamming distance between 2 strings"""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def all_genotypes(N, AAs):
    """Fills the sequence space with all possible genotypes."""
    return np.array(list(itertools.product(AAs, repeat=N)))

def neighbors(sequence, sequence_space):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==1]

def custom_neighbors(sequence, sequence_space, d):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==d]

def genEpiNet(N, K):
    """Generates a random epistatic network for a sequence of length
    N with, on average, K connections"""
    return {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

def fitness_i(sequence, i, epi, mem):
    """Assigns a (random) fitness value to the ith amino acid that
    interacts with K other positions in a sequence, """
    #we use the epistasis network to work out what the relation is
    key = tuple(zip(epi[i], sequence[epi[i]]))
    #then, we assign a random number to this interaction
    if key not in mem:
        mem[key] = np.random.uniform(0, 1)
    return mem[key]


def fitness(sequence, epi, mem):
    """Obtains a fitness value for the entire sequence by summing
    over individual amino acids"""
    print(sequence)
    print(epi)
    print(mem)
    return np.mean([
        fitness_i(sequence, i, epi, mem) # ω_i
        for i in range(len(sequence))
    ])

def makeNK(N, K, AAs):
    """Make NK landscape with above parameters"""
    f_mem = {}
    epi_net = genEpiNet(N, K)
    sequenceSpace = all_genotypes(N,AAs)
    seqspace = [list(i) for i in list(sequenceSpace)]
    land = [(x,y) for x, y in zip(sequenceSpace, [fitness(i, epi=epi_net, mem=f_mem) for i in sequenceSpace])]
    return land, sequenceSpace, epi_net

if __name__ == "__main__":
    land_K2, seq, _ = makeNK(3,0,"AC")#DEFGHIKL")
    #seed = np.array([x for x in "AAAAA"])

    #subsets = {x : [] for x in range(6)}

    #for seq in land_K2:
        #subsets[hamming(seq[0],seed)].append(seq)
