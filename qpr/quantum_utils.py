from pathlib import Path
import pickle
import numpy as np
import minorminer


def longest_chain_in_embed(e):
    return np.max([len(i) for i in e.values()])


def find_embedding_minorminer(Q, A, num_tries=100, print_output_flag = False):
    best_embedding = None
    best_chain_len = np.inf

    step = int(np.ceil(num_tries / 10))
    for i in range(num_tries):
        if print_output_flag is True:
            if i % step == 0:
                print(f'try {i+1} / {num_tries}')

        e = minorminer.find_embedding(Q, A)
        if e:  # to guarantee an embedding is produced
            chain_len = longest_chain_in_embed(e)
            if chain_len < best_chain_len:
                best_embedding = e
                best_chain_len = chain_len

    return best_embedding, best_chain_len


def cached_find_embedding(Q, A, qpu_id, probname, num_tries=100, hurry=False):
    '''
        qpu_id: e.g. 'DW_2000Q_5', DWaveSampler().solver.id
        probname: e.g. '3x3Grid_1'
    '''

    # UPDATE THIS WHEN YOU MAKE BACKWARD INCOMPATIBLE TO THIS FUNCTION!
    VERSION = 0

    flist = list(Path('.').glob(f'{qpu_id}__{probname}__chainlen*.pickle'))
    # We keep the best embedding only for now
    assert len(flist) < 2

    valid = False
    if flist:
        print('pickled data found')
        with open(flist[0], 'rb') as f:
            pickle_data = pickle.load(f)

        old_embedding, old_chain_len, old_Q, old_A = pickle_data[:4]
        old_probname, old_qpu_id, version = pickle_data[4:]

        if A is not None:
            assert A == old_A
        if Q is not None:
            assert Q.keys() == old_Q.keys()
        valid = (
            VERSION == version and probname == old_probname and
            qpu_id == old_qpu_id
        )

        if not valid:
            print('invalid pickle')
            new_file = 'invalid__' + str(flist[0])
            assert not Path(new_file).exists()
            flist[0].rename(new_file)

    if hurry and valid:
        print('skipping a new embedding attemp')
        return old_probname, old_chain_len

    # now that we're not in a rush, try the embedding one more time!
    new_embedding, new_chain_len = find_embedding_minorminer(
        Q, A, num_tries=num_tries
    )
    if not flist or new_chain_len < old_chain_len:
        pickle_data = (
            new_embedding, new_chain_len, Q, A, probname, qpu_id, VERSION
        )

        new_fname = f'{qpu_id}__{probname}__chainlen{new_chain_len}.pickle'
        if flist:
            print(f'Updating the chain length in {flist[0]}')
            flist[0].rename(new_fname)

        with open(new_fname, 'wb') as f:
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'pickled new data, chain length {new_chain_len}')

        return new_embedding, new_chain_len

    print(f'Returning the pickled data (new chain length {new_chain_len})')
    return old_embedding, old_chain_len


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
