import numpy as np
import torch

default_char_encoding={x : y for (x,y) in zip("ACDEFGHIKLMNPQRSTVWY",range(20))}

def map_str(string,char_encoding=default_char_encoding):
    """
    Maps
    """

    hold = torch.Tensor([char_encoding[x] for x in string])
    hold = hold.long()
    return hold

def collapse(array):
    """
    This needs to be changed, it is still iterative. Should still be faster though
    It functions by taking the first column of the array and then joining all subsequent ones to it.

    Collapses strings after their modification

    New is an array of the strings that need to be tested for their shannon entropy.

    Could be much faster if the operation to index shannon entropy is vectorized. There should be a way to do this.
    """
    new = array[:,0].copy()
    for i in range(1,array.shape[1]):
        new = np.char.add(new,array[:,i])
    return np.char.decode(new,"utf 8")

def generate_sequences(seed_sequence,num_mutations,amino_acids=list("ACDEFGHIKLMNPQRSTVWY")):

    """
    Generates all possible mutants from a given seed sequence with up to N concurrent mutations.

    Requires a seed sequence, a number of mutations, and an amino acid list which is provided as the 20 canonical by default
    """
    if type(amino_acids) != list:
        amino_acids = list(amino_acids)

    block_size = len(amino_acids)**num_mutations

    positions = list(combinations(range(len(seed_sequence)),num_mutations))

    print("Executing this code will generate {} sequences\n".format(block_size*len(positions)))
    print("If each sequence takes a millionth of a second, this will take {} seconds".format((block_size*len(positions))/1000000))

    query =  input("Do you want to proceed? y/[n]")
    if query == "y" or query == "Y":
        # Hold takes each block and is then reshaped at the end to produce a list of sequences
        hold = np.chararray((len(positions),block_size,len(seed_sequence)))

        """
        Generates the block for the first time
        """
        block = np.chararray((block_size,len(seed_sequence)))

        for i,char in enumerate(seed_sequence):
            block[:,i] = char

        """
        Iterates over the provided positions and generates a block of modified sequences for each position, then assigns
        them to a position in hold
        """
        for i,pos in enumerate(positions):

            current_block = block.copy()
            indexes = []

            # Generates arrays filled with each index
            for idx in pos:
                indexes.append(np.full(block_size,idx))

            # Generates all array modifications. The base is a repeated list of amino acid sequences as long as the block
            modifications = []
            modifications.append(np.array([amino_acids for x in range(int(block_size/len(amino_acids)))]).reshape(block_size,))

            # Generates all higher level amino acid lists, withe the first being A*20,C*20... and the second being
            # A*400, C*400 and so on
            for j,additional in enumerate(pos[1:]):
                j += 1
                modifications.append(np.array([list(x*int((len(amino_acids)**(j)))) for x in amino_acids]*int((block_size/((len(amino_acids))**(j+1))))).reshape(block_size,))

            # Goes over each mutation array and index array and reassigns the code in the current block
            for k in range(num_mutations):
                current_block[np.array([x for x in range(block_size)]),indexes[k]] = modifications[k]

            # Places the current block into the hold array
            hold[i] = current_block

        # Reshapes the hold array into a full list of sequences ready to be collapsed.
        hold = hold.reshape(len(positions)*block_size,len(seed_sequence))

    else:
        print("Process aborted")

    return hold
