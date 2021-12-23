import numpy as np
from fitness.fitness import Fitness, var_dict
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split


class Reconstruction(Fitness):
    def __init__(self, data, num_trees, seed, hidden):
        super().__init__(num_trees, seed)
        self.hidden = hidden
        print("Hidden Layer Sizes: {}".format(self.hidden))

    # with train/test split
    # def eval_embedding(self, embedding):
    #     y = var_dict['data_T'].T
    #     X_train, X_test, y_train, y_test = train_test_split(embedding, y, test_size=0.33, random_state=1)
    #     regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    #     return mse(regr.predict(X_test), y_test),

    # without split
    def eval_embedding(self, embedding):
        regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=self.hidden)\
            .fit(embedding, var_dict['data_T'].T)
        return mse(regr.predict(embedding), var_dict['data_T'].T),

