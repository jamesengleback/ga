import random
from multiprocessing.pool import ThreadPool, Pool
from concurrent.futures import ThreadPoolExecutor

AAS = list('ACDEFGHIKLMNPQRSTVWY')

def random_seq(n):
    return ''.join(random.choices(AAS,k=n))

def mutate(seq, vocab=AAS):
    seq = list(seq)
    seq[random.randint(0, len(seq)-1)] = random.choice(vocab)
    return ''.join(seq)

def crossover(a,b):
    cut = random.randint(0,min(len(a),len(b)))
    return random.choice([a[:cut] + b[cut:], b[:cut] + a[cut:]])

def eval(gene_pool, fn, n_process = None):
    if n_process is None:
        n_process = len(gene_pool)
    with ThreadPoolExecutor(n_process) as process_pool :
        results = process_pool.map(fn, gene_pool)
    return dict(zip(gene_pool,results))


