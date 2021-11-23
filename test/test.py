import time
import random
from string import ascii_lowercase
from pprint import pprint

import ga

def fn(gene):
    return sum(map(lambda a : a == 'a', gene))

def main():
    #gene_pool = [ga.random_seq(10, vocab=ascii_lowercase) for _ in range(10)]
    #pprint(ga.eval(gene_pool,fn))
    gene_pool = [ga.random_seq(32, vocab=ascii_lowercase) for _ in  range(10)]
    mutant = gene_pool[0]
    #mutant.apply(lambda gene : ga.random_mutate(gene, vocab=ascii_lowercase))
    #gene_pool.apply(ga.random_mutate)

    seq = ga.Sequential(
                        ga.Mutate(3,'*'),
                        ga.RandomMutate(),
                        ga.CrossOver(4),
                        ga.Evaluate(fn),
                        ga.Print(),
                        ga.Tournament(),
                        )    
    #print(seq(mutant))
    s = seq(gene_pool)
    print(s, type(s))


if __name__ == '__main__':
    main()
