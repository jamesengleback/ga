import unittest
import random
from string import ascii_lowercase, ascii_uppercase
import ga

class Test_random_seq(unittest.TestCase):
    def test_random_seq(self):
        from ga import random_seq
        N=8
        seq = random_seq(N, vocab=ascii_lowercase)
        assert len(seq) == N
        assert isinstance(seq,str)
        assert len(seq) == N

class Test_mutate(unittest.TestCase):
    def test_mutate(self):
        from ga import mutate
        from ga import random_seq
        import random
        N=8
        seq = random_seq(N, vocab=ascii_lowercase)
        seq_ = mutate(seq, random.randint(0,N-1), random.choice(ascii_uppercase))
        assert isinstance(seq, str)
        assert len(seq) == len(seq_)
        assert sum([i!=j for i, j in zip(seq, seq_)]) == 1

class Test_hamming(unittest.TestCase):
    def test_hamming(self):
        from ga import hamming
        assert hamming('aaa','aba') == 1
        assert hamming('aaa','abb') == 2

class Test_random_mutate(unittest.TestCase):
    def test_random_mutate(self):
        from ga import random_mutate
        s = ascii_lowercase
        s_ = random_mutate(s, vocab=ascii_uppercase)
        assert len(s) == len(s_)
        assert sum([i.isupper() for i in s_]) == 1
        assert sum([i in ascii_uppercase for i in s_]) == 1

class Test_crossover(unittest.TestCase):
    def test_crossover(self):
        from ga import crossover
        s = crossover(ascii_uppercase, ascii_lowercase)
        assert len(s) == len(ascii_uppercase)
        assert sum([i.isupper() for i in s]) >= 1
        assert sum([i.islower() for i in s]) >= 1

class Test_evaluate(unittest.TestCase):
    def test_evaluate(self):
        import time
        from ga import evaluate
        def fn(s):
            assert isinstance(s, str)
            time.sleep(0.1)
            return sum([i == 'a' for i in s])
        pop = ['big', 'time', 'charley', 'potatoes']
        _, result = evaluate(pop, fn)
        assert len(pop) == len(result)


class TestLayer(unittest.TestCase):
    def forward(self, obj, pop=None, n=8,l=8, *args):
        if pop is None:
            from ga import random_seq
            pop = [random_seq(l, ascii_lowercase) for _ in range(n)]
        pop_ = obj(pop, *args)
        if isinstance(pop_, tuple):
            pop, pop_ = pop_
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)
        return pop_, pop



class TestPrint(TestLayer):
    def testPrint(self):
        from ga import Print
        obj = Print()
        pop = ['big', 'time', 'charley', 'potatoes']
        self.forward(obj, pop)

class TestMutate(TestLayer):
    def testMutate(self):
        from ga import Mutate
        from ga import random_seq, hamming
        N=8
        L=8
        pos = random.randint(0,N-1)
        new = random.choice(ascii_uppercase)
        obj = Mutate(pos, new)
        pop, pop_ = self.forward(obj)
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)
        for i, j in zip(pop, pop_):
            assert len(i) == len(j)
            assert i != j

class TestRandomMutate(TestLayer):
    def testRandomMutate(self):
        from ga import RandomMutate
        from ga import random_seq
        obj = RandomMutate()
        pop, pop_ = self.forward(obj)
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)
        for i, j in zip(pop, pop_):
            assert len(i) == len(j)
            assert i != j


class TestCrossOver(TestLayer):
    def testCrossOver(self):
        from ga import CrossOver
        obj = CrossOver()
        pop, pop_ = self.forward(obj)
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)

class TestConstrained(TestLayer):
    def testConstrained(self):
        from ga import Constrained, Sequential, RandomMutate
        obj = Constrained(layers=Sequential(RandomMutate(vocab=ascii_lowercase)),
                fn=(fn:=lambda mutant : sum([i == 'a' for i in mutant])),
                thresh = lambda pop : max(map(fn,pop)) <=4,
                )

        pop, pop_ = self.forward(obj)
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)
        assert len(pop) == len(pop_)

class TestEvaluate(TestLayer):
    def testEvaluate(self):
        from ga import Evaluate
        import time
        def fn(s):
            assert isinstance(s, str)
            time.sleep(0.1)
            return sum([i == 'a' for i in s])
        obj = Evaluate(fn)
        pop, result = self.forward(obj)
        assert len(pop) == len(result)
        assert isinstance(result, list)

class TestSelectionLayer(unittest.TestCase):
    def forward(self, pop=None, n=8,l=8, *args):
        from ga import Evaluate
        import time
        if pop is None:
            from ga import random_seq
            pop = [random_seq(l, ascii_lowercase) for _ in range(n)]
        def fn(s):
            assert isinstance(s, str)
            time.sleep(0.1)
            return sum([i == 'a' for i in s])
        obj = Evaluate(fn)
        pop, result = obj(pop)
        return pop, result


class TestTournament(TestSelectionLayer):
    def testTournament(self):
        from ga import Tournament
        obj = Tournament()
        pop, result = self.forward()
        pop_ = obj((pop, result))
        assert len(pop_) < len(pop)


class TestPickTop(TestSelectionLayer):
    def testPickTop(self):
        from ga import PickTop
        obj = PickTop()
        pop, result = self.forward()
        pop, pop_ = obj((pop, result))
        assert len(pop) > len(pop_)
        assert isinstance(pop_, list)

class TestPickBottom(TestSelectionLayer):
    def testPickBottom(self):
        from ga import PickBottom
        obj = PickBottom()
        pop, result = self.forward()
        pop, pop_ = obj((pop, result))
        assert len(pop) > len(pop_)
        assert isinstance(pop_, list)

class TestClone(TestLayer):
    def testClone(self):
        from ga import Clone
        N=8
        obj = Clone(N)
        pop, pop_ = self.forward(obj)
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)
        assert len(pop) == len(pop_)

class TestSequential(TestLayer):
    def test_sequential(self):
        fn = lambda s : sum([i == 'a' for i in s])
        POP_SIZE = 8
        contrained = ga.Constrained(\
                        fn=fn,
                        layers=ga.Sequential(\
                                             ga.RandomMutate(),
                                             ga.CrossOver(),
                                             ),
                        thresh=lambda p : max(map(fn,p)) < 4,
                        )
        sequential = ga.Sequential(\
                        contrained,
                        ga.Evaluate(fn),
                        ga.Tournament(),
                        ga.Clone(POP_SIZE),
                        )
        pop = [''.join(random.choices(ascii_lowercase,k=8)) for _ in range(POP_SIZE)]
        pop_ = sequential(pop)
        #pop, pop_ = self.forward(obj)
        assert len(pop) == len(pop_)
        assert isinstance(pop_, list)
        assert len(pop) == len(pop_)
        assert len(pop) == len(pop_)
