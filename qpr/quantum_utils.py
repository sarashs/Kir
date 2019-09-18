import numpy as np
import minorminer


def longest_chain_in_embed(e):
    return np.max([len(i) for i in e.values()])


def find_embedding_minorminer(Q, A, num_tries=100):
    best_embedding = None
    best_chain_len = np.inf

    for i in range(num_tries):
        e = minorminer.find_embedding(Q, A)
        if e:  # to guarantee an embedding is produced
            chain_len = longest_chain_in_embed(e)
            if chain_len < best_chain_len:
                best_embedding = e
                best_chain_len = chain_len

    return best_embedding, best_chain_len
