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

#def undrift(seq, fn, condition):
#    done=False
#    while not done:
#        seq_ = random_mutate(seq)
#        if (a:=fn(seq_)) > (b:=fn(seq)):
#            seq = seq_
#        done = condition(seq)
#        print(seq, a, b)
#    return seq

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

######====================

#class Mutant:
#    def __init__(self,
#                gene,
#                **kwargs,
#                ):
#        self.gene = gene
#        self.__dict__ = {**self.__dict__, **kwargs}
#    def mutate(self, pos:int, new):
#        '''
#        change self.gene at pos to new
#        '''
#        self.gene = mutate(self.gene, pos, new)
#    def apply(self, fn):
#        '''
#        apply function to self.gene
#        '''
#        self.gene = fn(self.gene)
#    def __len__(self):
#        return len(self.gene)
#    def __getitem__(self, idx):
#        return self.gene[idx]
#    def __setitem__(self, pos, new):
#        self.gene = mutate(self.gene, pos, new)
#    def __repr__(self):
#        return f"ga.Mutant: [{' '.join([f'{i}: {j}' for i,j in zip(self.__dict__.keys(), self.__dict__.values())])}]"
#    def __str__(self):
#        return self.gene
#
#class Pool: # if not better than a list then delete
#    def __init__(self,
#                *args,
#                ):
#        self.mutants = list(args)
#    def apply(self, fn): # for applying fns to self.mutants
#        return evaluate(self.mutants, fn)
#    def __len__(self):
#        return len(self.mutants)
#    def __getitem__(self, idx):
#        return self.mutants[idx]
#    def __repr__(self):
#        newline='\n'
#        return f"{newline.join([str(i) for i in self.mutants])}"

#class Layer:
#    def __init__(self, fn):
#        self.fn = fn
#    def __call__(self, pop):
#        return self.fn(pop)

class Print:
    def __init__(self):
        pass
    def __call__(self, x):
        assert isinstance(x, list)
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
        return x
    def __str__(self):
        return 'Mutate'
        #return list(map(lambda x : mutate(x, 2,'s'), x))
        #return list(map(lambda x_ : mutate(x_, self.pos, self.new), x))

class RandomMutate:
    def __init__(self):
        pass
    def __call__(self, x):
        #return [random_mutate(i) for i in x]
        return x
    def __str__(self):
        return 'RandomMutate'

class CrossOver:
    def __init__(self, n=None):
        self.n=n
    def __call__(self, pop):
        n = range(len(pop)) if self.n is None else range(self.n)
        return [crossover(*random.choices(pop,k=2)) for _ in n]
    def __str__(self):
        return 'CrossOver'

class Evaluate:
    # returns dict 
    def __init__(self, fn_):
        self.fn_ = fn_
    def __call__(self,x):
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

