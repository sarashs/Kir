import numpy as np
import minorminer


def longest_chain_in_embed(e):
    return np.max([len(i) for i in e.values()])


def find_embedding_minorminer(Q, A, num_tries=100):
    best_embedding = None
    best_chain_len = np.inf

    step = int(np.ceil(num_tries / 10))
    for i in range(num_tries):
        if i % step == 0:
            print(f'try {i+1} / {num_tries}')

        e = minorminer.find_embedding(Q, A)
        if e:  # to guarantee an embedding is produced
            chain_len = longest_chain_in_embed(e)
            if chain_len < best_chain_len:
                best_embedding = e
                best_chain_len = chain_len

    return best_embedding, best_chain_len


def get_all_min_energy(sample_set):
    '''
        .data() sorts by energy by defaults but returns an iterator (not a
                SampleSet) the iterator yields a named tuple
        .samples(n) sort by energy, take at most n samples, return a
                    SampleArray which is a view, mapping the var names
                    to the values (i.e returns dicts), It is indexable
                    i.e. .samples()[10] works
        .record returns record array of Sample objects which is basically a
                numpy-scliceable list of named tuples (samples).
                Also .record.energy returns a numpy array of energies,
                .record.samples returns a 2d numpy array of qubo answers etc.
        Iterating over the SampleSet, calls .samples() internally, i.e. it
                  gets sorted
        .first calls data() internally so it does the sorting anyway!

        This function returns all the min energy solutions as a list of
        {var name: val} dicts
    '''

    min_energy = np.min(sample_set.record.energy)
    # use .record since it is slicing friendly, this returns a 2d-array-like
    # recarray
    records = sample_set.record[sample_set.record.energy == min_energy]
    # make dicts out of each answer using the original var names
    # (i.e. sample_set.variables)
    return (
        [dict(zip(sample_set.variables, i.sample)) for i in records],
        min_energy
    )
