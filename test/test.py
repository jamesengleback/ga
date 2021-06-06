import time
import random

import ga

def fn(gene):
    time.sleep(0.5)
    return random.randint(0,10)

def main():
    gene_pool = [ga.random_seq(10) for i in range(10)]
    print(ga.eval(gene_pool,fn))
    


if __name__ == '__main__':
    main()
