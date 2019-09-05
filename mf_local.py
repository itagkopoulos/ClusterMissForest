from rfimpute import MissForestImputation
from randomforest import RandomForest
import numpy as np 

class MissForestImputationLocal(MissForestImputation):

    # def __init__(self, max_iter, init_imp, n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
    #              max_features, max_leaf_nodes, 
    #              min_impurity_decrease, bootstrap, 
    #              random_state, verbose, warm_start, 
    #              class_weight, n_cores):
    #     super().__init__(max_iter, init_imp)

    #     self.n_estimators = n_estimators
    #     self.max_depth = max_depth
    #     self.min_samples_split = min_samples_split
    #     self.min_samples_leaf = min_samples_leaf
    #     self.min_weight_fraction_leaf = min_weight_fraction_leaf
    #     self.max_features = max_features
    #     self.max_leaf_nodes = max_leaf_nodes
    #     self.min_impurity_decrease = min_impurity_decrease
    #     self.bootstrap = bootstrap
    #     self.random_state = random_state
    #     self.verbose = verbose
    #     self.warm_start = warm_start
    #     self.class_weight = class_weight
    #     self.n_cores = n_cores
    def __init__(self, mf_params, rf_params):
        super().__init__(**mf_params)
        self.class_weight = rf_params.pop('class_weight')
        self.params = rf_params

    # def get_params(self, vt):
    #     """ return parameters of RandomForest """
    #     params = None

    #     if vt == 'numerical':
    #         params = {
    #             'n_estimators' : self.n_estimators,
    #             'criterion' : 'mse',
    #             'max_depth' : self.max_depth,
    #             'min_samples_split' : self.min_samples_split,
    #             'min_samples_leaf' : self.min_samples_leaf,
    #             'min_weight_fraction_leaf' : self.min_weight_fraction_leaf,
    #             'max_features' : self.max_features,
    #             'max_leaf_nodes' : self.max_leaf_nodes,
    #             'min_impurity_decrease' : self.min_impurity_decrease,
    #             'min_impurity_split' : None,
    #             'bootstrap' : self.bootstrap,
    #             'oob_score' : False,
    #             'n_jobs' : self.n_cores,
    #             'random_state' : self.random_state,
    #             'verbose' : self.verbose,
    #             'warm_start' : self.warm_start,
    #         }
    #     else:
    #         params = {
    #             'n_estimators' : self.n_estimators,
    #             'criterion' : 'gini',
    #             'max_depth' : self.max_depth,
    #             'min_samples_split' : self.min_samples_split,
    #             'min_samples_leaf' : self.min_samples_leaf,
    #             'min_weight_fraction_leaf' : self.min_weight_fraction_leaf,
    #             'max_features' : self.max_features,
    #             'max_leaf_nodes' : self.max_leaf_nodes,
    #             'min_impurity_decrease' : self.min_impurity_decrease,
    #             'min_impurity_split' : None,
    #             'bootstrap' : self.bootstrap,
    #             'oob_score' : False,
    #             'n_jobs' : self.n_cores,
    #             'random_state' : self.random_state,
    #             'verbose' : self.verbose,
    #             'warm_start' : self.warm_start,
    #             'class_weight' : self.class_weight,
    #         }

    #     return params 

    def miss_forest_imputation(self, matrix_for_impute):
        self.matrix_for_impute = matrix_for_impute
        self.initial_guess()

        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1
        
        while True:
            if cur_iter > self.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return
            print("Iteration " + str(cur_iter))

            for var in self.vari:
                cur_X = self.cur_iter_matrix
                _, p = np.shape(cur_X)
                cur_obsi = self.obsi[var]
                cur_misi = self.misi[var]
                if (len(cur_misi) == 0):
                    continue

                p_train = np.delete(np.arange(p), var)
                X_train = cur_X[cur_obsi, :][:, p_train]
                y_train = cur_X[cur_obsi, :][:, var]
                X_test = cur_X[cur_misi, :][:, p_train]
                
                rf = RandomForest(self.params, self.class_weight)
                imp = rf.fit_predict(X_train, y_train, X_test, self.vart[var])
                self.cur_iter_matrix[cur_misi, var] = imp

            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return

            #Update the previous_iter_matrix
            self.previous_iter_matrix = np.copy(self.cur_iter_matrix)
            cur_iter = cur_iter + 1
