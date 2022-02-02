import argparse
import csv
import os
from read_data import read_data
from numpy import random
from main import MultiTreeGP
from eval import eval_embedding
import numpy as np


class RunData(object):
    def __init__(self):
        self.elitism = 1

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def init_data(rd):
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="Integer for seeding random generator", type=int)
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="wine")
    parser.add_argument("--dir", help="Dataset directory", type=str, default="../data")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int, default=1000)
    parser.add_argument("-p", "--pop", help="Size of population", type=int, default=100)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("--erc", dest='use_ercs', help="Use ephemeral random constants", action='store_true')
    parser.add_argument("--parsimony", help="use parsimony pressure", dest='use_parsimony', action='store_true')
    parser.add_argument("--dim", dest="n_dims", type=int, default=2)
    parser.add_argument("-m", "--measure", help="Measure to be used for fitness", type=str, default="spearmans",
                        choices=["spearmans", "mse", "nrmse", "pearsons", "umap_cost"])
    parser.add_argument("-nn", "--nearest_neighbors", help="Nearest neighbors to be used for UMAP cost", type=int,
                        default=15)
    parser.add_argument("-f", "--fitness", dest="fitness", help="Name of fitness function to use", type=str, default='reconstruction')
    parser.add_argument("-hl", "--hidden", dest="hidden", type=str, default="100")

    parser.set_defaults(use_parsimony=False)
    parser.set_defaults(use_ercs=False)

    args = parser.parse_args()
    print(args)
    update_experiment_data(rd, args)

    hidden = []

    for layer in rd.hidden.split("-"):
        hidden.append(int(layer))

    rd.hidden = hidden

    rd.all_data = read_data("%s/%s.data" % (args.dir, args.dataset))
    rd.data = rd.all_data["data"]
    rd.labels = rd.all_data["labels"]
    rd.num_instances = rd.data.shape[0]
    rd.num_features = rd.data.shape[1]
    rd.num_classes = len(set(rd.labels))


def update_experiment_data(data, ns):
    dict = vars(ns)
    for i in dict:
        setattr(data, i, dict[i])
        # data[i] = dict[i]


def write_ind_to_file(ind, file_prefix):

    line_list = []

    # add constructed features to lines

    for cf in [str(tree) for tree in ind]:
        line_list.append(cf + "\n")

    line_list.append("\n")

    fname = "{}/{}_ind.txt".format(rd.outdir, file_prefix)
    if not os.path.exists(fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    fl = open(fname, 'w')
    fl.writelines(line_list)
    fl.close()


def write_embedding_to_file(embedding, file_prefix):

    fname = "{}/{}_emb.data".format(rd.outdir, file_prefix)
    if not os.path.exists(fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    fl = open(fname, 'w')
    fl.write("classLast,{},{},comma".format(rd.n_dims, rd.num_classes))
    lines = []
    for index, instance in enumerate(embedding):
        line = np.append(instance, int(rd.labels[index])).tolist()
        line[-1] = int(line[-1])
        lines.append("\n"+",".join(map(str, line)))
    fl.writelines(lines)
    fl.close()


rd = RunData()


def main():
    random.seed(rd.seed)
    auto_encoder = MultiTreeGP(data=rd.data, fitness=rd.fitness, n_dims=rd.n_dims,
                               seed=rd.seed, gens=rd.gens, popsize=rd.pop, hidden=rd.hidden)
    auto_encoder.run()

    embedding = auto_encoder.embedding
    acc = eval_embedding(embedding, rd.labels)

    results = {"accuracy": acc, "time": auto_encoder.time, "size": len(auto_encoder.best)}

    write_ind_to_file(auto_encoder.best, str(rd.seed))
    write_embedding_to_file(embedding, str(rd.seed))

    csv_file = "{}/{}_results.txt".format(rd.outdir, rd.seed)
    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)


if __name__ == '__main__':
    init_data(rd)
    main()
