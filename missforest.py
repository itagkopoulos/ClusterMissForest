from mf_local import MissForestImputationLocal
from mf_slurm import MissForestImputationSlurm
from time import time
import os, shutil
import numpy as np

InitImpOptions = ['mean', 'zero', 'knn']
ParallelOptions = ['slurm', 'local']

class MissForest:

    def __init__(self, max_iter=10, init_imp='mean',
                 n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                 max_features='sqrt', max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, bootstrap=True, 
                 random_state=None, verbose=0, warm_start=False, 
                 class_weight=None, 
                 partition=None, n_cores=1, n_nodes=1, 
                 node_features=1, memory=2000, time='1:00:00', parallel='local'):
        # MissForest parameters
        self.max_iter = max_iter
        self.init_imp = init_imp

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        
        self.partition = partition
        self.n_cores = n_cores
        self.n_nodes = n_nodes 
        self.node_features = node_features
        self.memory = memory
        self.time = time

        self.parallel = parallel

    def get_mf_params(self):
        """ return parameters of MissForest """
        return {
            'max_iter' : self.max_iter,
            'init_imp' : self.init_imp,
            'vart' : self.vart,
        }

    def get_rf_params(self):
        """ return parameters of RandomForest """
        return {
            'n_estimators' : self.n_estimators,
            'max_depth' : self.max_depth,
            'min_samples_split' : self.min_samples_split,
            'min_samples_leaf' : self.min_samples_leaf,
            'min_weight_fraction_leaf' : self.min_weight_fraction_leaf,
            'max_features' : self.max_features,
            'max_leaf_nodes' : self.max_leaf_nodes,
            'min_impurity_decrease' : self.min_impurity_decrease,
            'bootstrap' : self.bootstrap,
            'n_jobs' : self.n_cores,
            'random_state' : self.random_state,
            'verbose' : self.verbose,
            'warm_start' : self.warm_start,
            'class_weight' : self.class_weight,
        }

    def get_slurm_params(self):
        return {
            'partition' : self.partition,
            'n_nodes' : self.n_nodes,
            'n_cores' : self.n_cores,
            'node_features' : self.node_features,
            'memory' : self.memory,
            'time' : self.time,
        }
        

    def _check_inputs(self, xmis):
        """ private method, validating all inputs """
        self.vart = []

        try:
            n, p = np.shape(xmis)
            for v in range(p):
                if np.issubdtype(xmis[:, v].dtype, np.number):
                    self.vart.append(1) # numerical
                else:
                    self.vart.append(0) # categorical
        except:
            raise ValueError("xmis: not a matrix")


    #     if self.parallel not in ParallelOptions:
    #         raise ValueError("parallel: not one of slurm, local")
    #     if self.parallel == 'slurm':
    #         if type(self.n_nodes) != int or self.n_nodes < 1:
    #             raise ValueError("n_nodes: not a positive integer")
    #         elif self.n_nodes > p:
    #             raise ValueError("n_nodes: nodes should be less than variables of dataset")
    #         if type(self.node_features) != int or self.node_features < 1:
    #             raise ValueError("node_features: not a positive integer")
    #         if type(self.memory) != int or self.memory < 1:
    #             raise ValueError("memory: not a positive integer")
    #     if type(self.max_iter) != int or self.max_iter < 1:
    #         raise ValueError("max_iter: not a positive integer")
    #     if self.init_imp not in InitImpOptions:
    #         raise ValueError("init_imp: not one of mean, zero, knn")
    #     if type(self.n_cores) != int or self.n_cores < 1:
    #         raise ValueError("n_cores: not a positve integer")

    def _init_dirs(self):
        """ private method, initialize hidden files """
        files = ['.out', '.err', '.dat']
        for file in files:
            if os.path.exists(file):
                shutil.rmtree(file)
            os.mkdir(file)

    def fit_transform(self, xmis):
        """ return imputed matrix-like data """
        self._check_inputs(xmis)
        mf_params = self.get_mf_params()
        rf_params = self.get_rf_params()
        sl_params = self.get_slurm_params()
        if self.parallel == 'local':
            mf = MissForestImputationLocal(mf_params, rf_params)
        else:
            self._init_dirs()
            mf = MissForestImputationSlurm(mf_params, rf_params, **sl_params)
        mf.miss_forest_imputation(xmis)

        return mf.result_matrix

def rmse(nmis, ximp, xtrue):
    rss = np.sum((xtrue - ximp) ** 2)
    rmse = np.sqrt(rss / nmis)

    return rmse

if __name__ == "__main__":

    print("reading data files ...")
    # data = np.loadtxt('data/data0.5_50.csv', delimiter = ',')
    # true_data = np.loadtxt('data/data0.5.csv', delimiter = ',')
    data = np.loadtxt('data/big_data50.csv', delimiter=',')
    true_data = np.loadtxt('data/big_data.txt', skiprows=1, usecols=tuple(np.arange(2, 4098)))
    nmis = len(np.argwhere(np.isnan(data)))
    n, p = np.shape(data)
    mf = MissForest(max_iter=5, init_imp='mean', parallel='slurm', n_nodes=4, n_cores=32, node_features=1024, memory=64000, time='1-00:00')
    start = time()
    ximp = mf.fit_transform(data) 
    duration = time() - start

    print(rmse(nmis, ximp, true_data))







