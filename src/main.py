from deap import gp, creator, base, tools
from scipy.special import expit
from fitness.fitness import Fitness, init_worker
from fitness.reconstruction import Reconstruction
from multiprocessing.sharedctypes import RawArray
from multiprocessing import Pool
from ea import ea
import numpy as np
import random
import time
import copy
import os, multiprocessing

import multiprocessing as mp
mp.set_start_method('fork')  # TODO: something with this making it work on Mac??

CXPB = 0.7
MUTPB = 0.3
ELITISM = 10

fitness_functions = {"reconstruction": Reconstruction}


class MultiTreeGP:

    def __init__(self, data, fitness: (str, Fitness), gens=100, popsize=100, seed: int = None,
                 n_dims: int = 2, functions=None, hidden=(100,)):
        self.gens = gens
        self.popsize = popsize
        self.data = data
        self.data_t = data.T
        self.time = 0
        self.use_ercs = True
        self.functions = functions
        self.n_dims = n_dims
        self.seed = seed
        self.hidden = hidden

        self.fitness = Reconstruction(self.data, self.n_dims, self.seed, self.hidden)

        self.best = None
        self.embedding = None
        self.stats = None

    def init_primitives(self, pset):

        # TODO: implement collection of functions

        if self.functions is None:  # default functions
            pset.addPrimitive(np.add, 2)
            pset.addPrimitive(np_many_add, 5)
            pset.addPrimitive(np.subtract, 2)
            pset.addPrimitive(np.multiply, 2)
            pset.addPrimitive(add_abs, 2)
            pset.addPrimitive(sub_abs, 2)
            pset.addPrimitive(protected_div, 2)
            pset.addPrimitive(np.maximum, 2)
            pset.addPrimitive(np.minimum, 2)
            pset.addPrimitive(relu, 1)
            pset.addPrimitive(sigmoid, 1)
            pset.addPrimitive(mt_if, 3)
        else:
            for function in self.functions:
                pset.addPrimitive(function[0], function[1])

        if self.use_ercs:
            pset.addEphemeralConstant("rand", ephemeral=lambda: random.uniform(-1, 1))

    def init_stats(self):
        """
        Initialises a MultiStatistics object to capture data.

        :return: the MultiStatistics object
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        return stats

    def init_toolbox(self, pset, n_trees):
        creator.create("Individual", list, fitness=creator.Fitness, pset=pset)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=6)
        self.toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, self.toolbox.expr)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.tree, n=n_trees)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)
        self.toolbox.register("mate", lim_xmate_aic)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=8)
        self.toolbox.register("mutate", lim_xmut, expr=self.toolbox.expr_mut)
        self.toolbox.register("select", tools.selTournament, tournsize=7)
        self.toolbox.register("evaluate", self.evaluate)

    def build_embedding(self, ind):
        no_instances = self.data.shape[0]
        no_trees = len(ind)
        feature_major = self.data.T
        # [no_trees x no_instances]
        # we do it this way so we can assign rows (constructed features) efficiently.
        result = np.zeros(shape=(no_trees, no_instances))
        for i, expr in enumerate(ind):
            func = self.toolbox.compile(expr=expr)
            vec = func(*feature_major)
            if (not isinstance(vec, np.ndarray)) or vec.ndim == 0:
                # it decided to just give us a constant back...
                vec = np.repeat(vec, no_instances)
            result[i] = vec

        return result.T

    def evaluate(self, ind):
        embedding = self.build_embedding(ind)
        return self.fitness.eval_embedding(embedding)

    def run(self):
        random.seed(self.seed)

        pset = gp.PrimitiveSet("MAIN", len(self.data[0]), prefix="f")
        pset.context["array"] = np.array
        self.init_primitives(pset)

        if self.fitness.min:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))  # minimise
        else:
            print('reached')
            creator.create("Fitness", base.Fitness, weights=(1.0,))  # maximise

        # set up toolbox
        self.toolbox = base.Toolbox()
        self.init_toolbox(pset, self.n_dims)

        parallel_data = self.fitness.init_parallel_data(self.data)

        # flatten all data in parallel dict, and store in buffer
        for key in parallel_data.keys():
            parallel_data[key] = flatten_data(parallel_data[key])

        threads = multiprocessing.cpu_count()
        print("Using " + str(threads) + " threads")
        with Pool(processes=threads, initializer=init_worker,
                  initargs=(parallel_data, )) as pool:
            self.toolbox.register("map", pool.map)
            start_time = time.time()
            pop, stats, hof, logbook = self.evolve()
            end_time = time.time()
            self.time = end_time - start_time
            self.best = hof[0]
            self.embedding = self.build_embedding(self.best)
            print('Main Thread Complete , Total Time Taken = {}'.format(end_time - start_time))

    def evolve(self):
        pop = self.toolbox.population(n=self.popsize)
        self.stats = self.init_stats()
        hof = tools.HallOfFame(1)  # we only want the best B-)
        pop, logbook = ea(pop, self.toolbox, CXPB, MUTPB, ELITISM, self.gens, self.fitness, self.stats,
                          halloffame=hof, verbose=True)
        return pop, self.stats, hof, logbook


def flatten_data(array):
    raw_array_shape = array.shape
    if array.ndim == 2:
        raw_array = RawArray('d', raw_array_shape[0] * raw_array_shape[1])
    else:
        raw_array = RawArray('d', raw_array_shape[0])
    raw_array_np = np.frombuffer(raw_array, dtype=np.float64).reshape(raw_array_shape)
    np.copyto(raw_array_np, array)

    return raw_array, raw_array_shape




###########################
# MULTI-TREE GP OPERATORS #
###########################

def maxheight(v):
    return max(i.height for i in v)

# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        #TODO: Unhardcode
        if maxheight(ind) > 9:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr,pset=ind.pset)
    ind[i1] = indx[0]
    return ind,


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    # print(res)
    return res


def mi_mut(ind, expr, build):
    embedding = build(ind)
    feature_wise = embedding.T
    num_trees = len(ind)

    # calculate MI between each embedding dimension
    embedding_mi = np.zeros((num_trees, num_trees))
    for i in range(num_trees-1):
        for j in range(i+1, num_trees):
            embedding_mi[i, j] = ee.mi(feature_wise[i], feature_wise[j])

    highest = np.argmax(embedding_mi, axis=1)  # identify tree pair with highest MI

    # select largest of tree pair
    if len(ind[highest[0]]) < len(ind[highest[1]]):
        ind_to_mut = highest[1]
    else:
        ind_to_mut = highest[0]

    # replace largest with new tree
    ind[ind_to_mut] = expr()

    return ind,


def xmate_aic(ind1, ind2):
    assert len(ind1) == len(ind2)
    for i in range(len(ind1)):
        ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def lim_xmate_aic(ind1, ind2):
    """
    Basically, keep only changes that obey max depth constraint on a tree-wise (NOT individual-wise) level.
    :param ind1:
    :param ind2:
    :return:
    """
    keep_inds = [copy.deepcopy(ind1), copy.deepcopy(ind2)]
    new_inds = list(xmate_aic(ind1, ind2))
    for i, ind in enumerate(new_inds):
        for j, tree in enumerate(ind):
            if tree.height > 9:
                new_inds[i][j] = keep_inds[i][j]
    return new_inds

################
# GP FUNCTIONS #
################

def add_abs(a, b):
    return np.abs(np.add(a, b))


def sub_abs(a, b):
    return np.abs(np.subtract(a, b))


def mt_if(a, b, c):
    return np.where(a < 0, b, c)


def sigmoid(x):
    return expit(x)


def relu(x):
    return x * (x > 0)


def np_many_add(a, b, c, d, e):
    return a + b + c + d + e


def analytic_quotient(a, b):
    return a / np.sqrt(1 + b**2)


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


