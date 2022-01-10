import random
import heapq
import json
#from multiprocessing.pool import ThreadPool, Pool
from concurrent.futures import ThreadPoolExecutor#, ProcessPoolExecutor

AAS = list('ACDEFGHIKLMNPQRSTVWY')

def random_seq(n : int, 
               vocab=AAS):
    ''' generates random string of length n from characters in vocab 
        (iterable returning strings)
    '''
    return ''.join(random.choices(vocab,k=n))

def mutate(seq, 
           pos:int, 
           new:str):
    ''' mutate string at pos to new
    '''
    seq = list(seq)
    seq[pos] = new
    return ''.join(seq)

def hamming(a:str,
            b:str):
    ''' return hamming distance between two strings of the same length
    '''
    assert len(a) == len(b)
    return sum([i!=j for i,j in zip(a,b)])

def random_mutate(seq:str, 
                  vocab=AAS, 
                  pos_weights=None, 
                  vocab_weights=None):
    ''' mutate string at random position to random characters
        from vocab (iterable of chars),
        seq : str 
        vocab : iterable returning strings
        pos_weights : iterable of floats, maps to seq
            probability weights for position selection
        vocab_weights : iterable of floats, maps to vocab 
            probability weights for substitution selection
    '''
    mxn_site = random.choices(range(len(seq)), weights=pos_weights, k=1)[0]
    new = random.choices(vocab, weights=vocab_weights, k=1)[0]
    return mutate(seq, mxn_site, new)

def crossover(a:str,
              b:str):
    ''' randomly splice two strings
        returns string
    '''
    cut = random.randint(1,min(len(a),len(b))-1)
    return random.choice([a[:cut] + b[cut:], b[:cut] + a[cut:]])

def evaluate(gene_pool, 
             fn, 
             **kwargs):
    ''' gene_pool : iterable (except generators)
        fn : function to map to gene_pool
        map a function to an iterable with multiprocessing 
        return original iterable (?generators) and list of 
        function evaluations

    '''
    #fn = lambda x : fn_(x,  **kwargs)
    with ThreadPoolExecutor(**kwargs) as process_pool :
        results = process_pool.map(fn, gene_pool)
    return gene_pool, list(results)

###
### Should i make a base class?

class Print:
    ''' print population and return population
    '''
    def __init__(self):
        pass
    def __call__(self, x):
        if isinstance(x, tuple):
            x, args = x
            for i,j in zip(x, args):
                print(i, j)
            return x, args
        else:
            for i in x:
                print(i)
            return x
    def __str__(self):
        return 'Print'

class Mutate:
    ''' calls ga.mutate
        __init__ args : pos : int , new:str
    '''
    def __init__(self, 
                 pos:int, 
                 new:int):
        self.pos=pos
        self.new=new
    def __call__(self, x):
        if isinstance(x, tuple):
            x, args = x
        if hasattr(x, '__len__'):
            x = [mutate(i, self.pos, self.new) for i in x]
        else:
            x = mutate(x, self.pos, self.new)
        return x
    def __str__(self):
        return 'Mutate'

class RandomMutate:
    ''' calls ga.random_mutate
        __init__ args : pos : int , new:str
    '''
    def __init__(self, 
                 vocab=AAS, 
                 pos_weights=None, 
                 vocab_weights=None):
        self.vocab = vocab
        self.pos_weights = pos_weights
        self.vocab_weights = vocab_weights
    def __call__(self, x):
        if isinstance(x, tuple):
            x, args = x
        return [random_mutate(i, vocab=self.vocab) for i in x]
    def __str__(self):
        return 'RandomMutate'

class CrossOver:
    ''' calls ga.crossover
        __init__ args : 
            n : int, default : None
                number of children to return, default: len(pop)
    '''
    def __init__(self, 
                 n=None):
        self.n=n
    def __call__(self, pop):
        if isinstance(pop, tuple):
            pop, args = pop
        n = len(pop) if self.n is None else self.n
        return [crossover(*random.choices(pop,k=2)) for _ in range(n)]
    def __str__(self):
        return 'CrossOver'

class Constrained:
    ''' repeat same sequence of mutation layers (ga.Sequential)
        until new population satisfies : 
            thresh(fn(pop)) == True
        __init__ args:
            layers : ga.Sequential
            fn     : function to evaluate - should return int or float
                     using monte carlo optimization here so there needs 
                     to be a number for better or worse
            thresh : fn - returns True when fn(pop) passes a threshold
                     criteria
    '''
    def __init__(self,
                 layers, # : ga.Sequential
                 fn,
                 thresh=None,
                 *args,
                 **kwargs,
                 ):
        self.layers = layers
        self.fn = fn
        self.thresh = thresh # fn
    def __call__(self, pop, n=1):
        ''' n : num iterations
        '''
        if self.thresh is not None:
            while not self.thresh(pop):
                pop = list(map(self.forward, pop))
        else:
            for _ in range(n):
                pop = self.forward(pop)
        return pop
    def forward(self, mutant):
        score0 = self.fn([mutant])
        done=False
        while not done:
            mutant_ =  self.layers([mutant])[0]
            score1 = self.fn(mutant_)
            if score1 < score0:
                done = True
        return mutant


class Evaluate:
    ''' calls ga.evaluate on pop
        __init__ args:
            fn_ : function to map to mutants
    '''
    # returns dict 
    def __init__(self, 
                 fn_, 
                 **kwargs):
        self.fn_ = fn_
        self.kwargs = kwargs
    def __call__(self,x):
        if isinstance(x, tuple):
            x, args = x
        return evaluate(x, self.fn_, **self.kwargs) # returns tuple
    def __str__(self):
        return 'Evaluate'

class Tournament:
    ''' Tournament selection for results tuple : (sequences, scores)
    '''
    def __init__(self, gt=True):
        self.gt = gt
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
        return [fitter(*random_pair()) for _ in range(len(pop_dict)//2)]
    def __str__(self):
        return 'Tournament'
    def forward(self, pop):
        f0 = [self.fn(i) for i in pop]
        pop_ = self.layers(pop)
        f1 = [self.fn(i) for i in pop_]
        return [i if j < k else l for i,j,k,l in zip(pop, f0, f1, pop_)]
    def __str__(self):
        return type(self).__name__
    def __repr__(self):
        return str(self)

class PickTop:
    '''
    '''
    def __init__(self, n=None, frac=2):
        self.n = n
        self.frac = frac
    def __call__(self, arg_tuple):
        if isinstance(arg_tuple, tuple):
            pop, scores = arg_tuple
            pop_dict = dict(zip(pop, scores))
            n = self.n if self.n is not None else len(pop) // self.frac
            return pop, heapq.nlargest(n, pop, key=lambda i : pop_dict[i])
        else:
            print(arg_tuple)
    def __str__(self):
        return 'PickBest'

class PickBottom:
    '''
    '''
    def __init__(self, n=None, frac=2):
        self.n = n
        self.frac = frac
    def __call__(self, arg_tuple):
        pop, scores = arg_tuple
        pop_dict = dict(zip(pop, scores))
        n = self.n if self.n is not None else len(pop)//self.frac
        return pop, heapq.nsmallest(n, pop, key=lambda i : pop_dict[i])
    def __str__(self):
        return 'PickBest'

class Clone:
    ''' Randomly sample n mutants (with replacement) from pop
        for up or down sampling
        __init__ args:
            n : int - number of mutants to return
    '''
    def __init__(self, n:int):
        self.n = n
    def __call__(self, x):
        return random.choices(x, k=self.n)
    def __str__(self):
        return 'Clone'

class Sequential:
    '''
    A pipeline of function modules to execute sequentially on
    pop in each call
    e.g
    >>> sequential = ga.Sequential(ga.RandomMutate(), 
                                   ga.CrossOver(),
                                   ...
                                   ga.Evaluate(<some function),
                                   ga.Tournament(),
                                   ga.Clone(),
                                   )
    >>> sequential(pop)
    >>> ... 
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

