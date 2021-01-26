def neighbors(sequence, landscape_dict, AAs='ACDEFGHIKLMNPQRSTVWY'):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    neighbors = generate_mutations(sequence, AAs=AAs)
    fits = []
    for i in neighbors:
        fits.append([i,landscape_dict[i]])
    return np.array(fits)

def generate_mutations(sequence, AAs='ACDEFGHIKLMNPQRSTVWY'):
    seqlist = list(sequence)
    muts = []
    #test = seqlist.copy()
    for ind, i in enumerate(seqlist):
        for amino in AAs:
            trial = seqlist.copy()
            trial[ind]=amino
            if (''.join(trial) not in muts) and ''.join(trial) != sequence:
                muts.append(''.join(trial))
    return muts

def is_maximum(seed, landscape, exp=False, AAs='ACDEFGHIKLMNPQRSTVWY'):
    if exp:
        h_neighbors   = generate_mutations(seed[0], AAs=AAs)
        #exp_neighbors = [seq for seq in h_neighbors if seq in landscape]
        n_data        = np.array([[x,landscape[x]] for x in h_neighbors if x in landscape])
    else:
        n_data = np.array(neighbors(seed[0], landscape, AAs=AAs))
   # print(n_data)
    fits = n_data[:,-1].astype(np.float)
    score = np.greater(seed[1],fits) #check if seed fitness is greater than neighbor fitnesses//
    if np.isin(False, score): #if there is a False (i.e. there  is a fitness greater than seed's), return False
        return False
    else:
        return True

def get_nmaxima(landscape_sub, landscape_dict, exp=False, AAs='ACDEFGHIKLMNPQRSTVWY'):
    s = time.time()
    n = 0
    as_array = [[x,y] for x, y in landscape_sub.items()]
    for i in as_array:
        out = is_maximum(i,landscape_dict, exp=exp, AAs=AAs)
        if out==True:
            n+=1
    e = time.time()
    print('Process time: {} seconds'.format(e-s))
    return n
