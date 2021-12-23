import numpy as np

var_dict = {}


class Fitness:
    def __init__(self, num_trees, seed):
        self.num_trees = num_trees
        self.seed = seed
        self.min = True

    def init_parallel_data(self, data):
        return {'data_T': data.T}

    @staticmethod
    def eval_embedding(self, dat_array):
        raise NotImplementedError("Evaluation function not implemented")

    def worker_func(self, func_str, tree_compiler):
        embedding = eval_trees(var_dict['data_T'], func_str, tree_compiler, self.num_trees)
        return self.eval_embedding(embedding)


def init_worker(raw_array_dict):
    for key in raw_array_dict.keys():
        var_dict[key] = np.frombuffer(raw_array_dict[key][0]).reshape(raw_array_dict[key][1])


def eval_trees(data_t, func_str, tree_compiler, num_trees):
    num_instances = data_t.shape[1]

    result = np.zeros(shape=(num_trees, num_instances))
    # TODO can this be matrixy?
    for i, f in enumerate(func_str):
        # Transform the tree expression in a callable function
        func = tree_compiler(expr=f)
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)

        result[i] = comp
        # result.append(comp)

    dat_array = result.T
    return dat_array
