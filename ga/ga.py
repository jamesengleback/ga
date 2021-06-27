import random
from multiprocessing.pool import ThreadPool

AAS = list('ACDEFGHIKLMNPQRSTVWY')

def random_seq(n):
    return ''.join(random.choices(AAS,k=n))

def mutate(seq, vocab=AAS, weights=None):
    seq = list(seq)
    nums = [i for i, j in enumerate(seq)]
    mxn_site = random.choices(nums, weights=weights, k=1)[0]
    seq[mxn_site] = random.choice(vocab)
    return ''.join(seq)

def crossover(a,b):
    cut = random.randint(0,min(len(a),len(b)))
    return random.choice([a[:cut] + b[cut:], b[:cut] + a[cut:]])

def eval(gene_pool, fn):
    with ThreadPool() as process_pool :
        results = process_pool.map(fn, gene_pool)
    process_pool.join()
    return dict(zip(gene_pool,results))


