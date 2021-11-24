import random
import heapq
import json
from multiprocessing.pool import ThreadPool, Pool
from concurrent.futures import ThreadPoolExecutor

AAS = list('ACDEFGHIKLMNPQRSTVWY')

def random_seq(n, vocab=AAS):
    return ''.join(random.choices(vocab,k=n))

def mutate(seq, pos:int, new):
    seq = list(seq)
    seq[pos] = new
    return ''.join(seq)

def hamming(a,b):
    assert len(a) == len(b)
    return sum([i!=j for i,j in zip(a,b)])

def random_mutate(seq, vocab=AAS, weights=None):
    mxn_site = random.choices(range(len(seq)), weights=weights, k=1)[0]
    new = random.choice(vocab)
    return mutate(seq, mxn_site, new)

def crossover(a,b):
    cut = random.randint(0,min(len(a),len(b)))
    return random.choice([a[:cut] + b[cut:], b[:cut] + a[cut:]])

def evaluate(gene_pool, fn_, n_process=None, *args, **kwargs):
    fn = lambda x : fn_(x, *args, **kwargs)
    if n_process is None:
        n_process = len(gene_pool)
    with ThreadPoolExecutor(n_process) as process_pool :
        results = process_pool.map(fn, gene_pool)
    return gene_pool, list(results)

class Print:
    def __init__(self):
        pass
    def __call__(self, x):
        if isinstance(x, tuple):
            x, args = x
            for i in x:
                print(i)
            return x, args
        else:
            for i in x:
                print(i)
            return x
    def __str__(self):
        return 'Print'

class Mutate:
    def __init__(self, pos, new):
        self.pos=pos
        self.new=new
    def __call__(self, x):
        if isinstance(x, tuple):
            x, args = x
        return x
    def __str__(self):
        return 'Mutate'

class RandomMutate:
    def __init__(self):
        pass
    def __call__(self, x):
        if isinstance(x, tuple):
            x, args = x
        return [random_mutate(i) for i in x]
        #return x
    def __str__(self):
        return 'RandomMutate'

class CrossOver:
    def __init__(self, n=None):
        self.n=n
    def __call__(self, pop):
        if isinstance(pop, tuple):
            pop, args = pop
        n = range(len(pop)) if self.n is None else range(self.n)
        return [crossover(*random.choices(pop,k=2)) for _ in n]
    def __str__(self):
        return 'CrossOver'

class Evaluate:
    # returns dict 
    def __init__(self, fn_):
        self.fn_ = fn_
    def __call__(self,x):
        if isinstance(x, tuple):
            x, args = x
        return evaluate(x, self.fn_) # returns tuple
    def __str__(self):
        return 'Evaluate'

class Tournament:
    '''
    Tournament selection for results tuple : (sequences, scores)
    '''
    def __init__(self, gt=True, frac=2):
        self.gt = gt
        self.frac = frac
    def __call__(self, arg_tuple):
        ## note - samples with replacement
        pop, scores = arg_tuple
        pop_dict = dict(zip(pop, scores))
        def random_choice():
            choice = random.choice(pop)
            pop.remove(choice)
            return choice
        random_pair = lambda : (random_choice(), random_choice())

        random_pair = lambda : random.choices(pop, k=2)
        key = lambda k : pop_dict[k]
        if self.gt:
            fitter = lambda a, b : a if key(a) > key(b) else b
        else:
            fitter = lambda a, b : a if key(a) < key(b) else b
        return [fitter(*random_pair()) for _ in range(len(pop_dict)//self.frac)]
    def __str__(self):
        return 'Tournament'

class PickTop:
    def __init__(self, n=None, frac=2):
        self.n = n
        self.frac = frac
    def __call__(self, arg_tuple):
        pop, scores = arg_tuple
        pop_dict = dict(zip(pop, scores))
        n = self.n if self.n is not None else len(pop)//self.frac
        return heapq.nlargest(n, pop, key=lambda i : pop_dict[i])
    def __str__(self):
        return 'PickBest'

class PickBottom:
    def __init__(self, n=None, frac=2):
        self.n = n
        self.frac = frac
    def __call__(self, arg_tuple):
        pop, scores = arg_tuple
        pop_dict = dict(zip(pop, scores))
        n = self.n if self.n is not None else len(pop)//self.frac
        return heapq.nsmallest(n, pop, key=lambda i : pop_dict[i])
    def __str__(self):
        return 'PickBest'

class Clone:
    def __init__(self, n=None):
        self.n = n
    def __call__(self, x):
        return random.choices(x, k=self.n)
    def __str__(self):
        return 'Clone'

class Sequential:
    '''
    A pipeline of functions.
    e.g
    ga.Sequential(ga.random_mutate, fn
    '''
    def __init__(self, *args, **kwargs):
        self.layers = list(args)
        self.__dict__ = {**self.__dict__, **kwargs}
        self.log = []
    def log_(self, **kwargs):
        self.log.append(kwargs)
    def savelog(self, path, mode='w'):
        with open(path, mode) as f:
            json.dump(self.log, f)
    def __call__(self, x):
        self.log_(layer=0, x=x)
        for fn in self.layers:
            x = fn(x) 
            self.log_(layer=str(fn), x=x)
        return x
    def __repr__(self):
        newline='\n'
        tab='\t'
        return f"ga.Sequential:{newline}{' -> '.join(list(map(lambda fn : str(fn), self.layers)))}"

