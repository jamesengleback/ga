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
    gene_pool = [ga.random_seq(32, vocab=ascii_lowercase) for _ in  range(32)]
    mutant = gene_pool[0]
    #mutant.apply(lambda gene : ga.random_mutate(gene, vocab=ascii_lowercase))
    #gene_pool.apply(ga.random_mutate)

    seq = ga.Sequential(
                        ga.Mutate(3,'*'),
                        ga.RandomMutate(),
                        ga.CrossOver(),
                        ga.Evaluate(fn),
                        ga.Tournament(frac=2),
                        ga.Clone(32),
                        ga.Evaluate(fn),
                        ga.Print(),
                        ga.ga.PickTop(frac=2),
                        ga.Print(),
                        ga.CrossOver(32),
                        ga.Print(),
                        ga.Evaluate(fn),
                        ga.Print(),
                        ga.ga.PickBottom(frac=2),
                        ga.Print(),
                        ga.CrossOver(32),
                        ga.Print(),
                        )    
    print(len(gene_pool))
    s = seq(gene_pool)
    print(len(s))
    #print(seq.log)
    seq.savelog('log')


if __name__ == '__main__':
    main()
