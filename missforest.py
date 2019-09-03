import os, shutil
import numpy as np
import rfimpute
from time import time

InitImpOptions = ['mean', 'zero', 'knn']
ParallelOptions = ['slurm', 'local']

class MissForest:

    def __init__(self, max_iter=10, init_imp='mean', parallel='local', n_nodes=1, 
                 n_cores=1, n_features=1, memory=None):
        self.max_iter = max_iter
        self.init_imp = init_imp
        self.parallel = parallel 
        self.n_nodes = n_nodes 
        self.n_cores = n_cores 
        self.n_features = n_features
        self.memory = memory

    def _check_inputs(self, xmis):
        """ private method, validating all inputs """
        try:
            n, p = np.shape(xmis)
        except:
            raise ValueError("xmis: not a matrix")

        if type(self.n_trees) != int or self.n_trees < 1:
            raise ValueError("n_trees: not a positive integer")

        if type(self.max_iter) != int or self.max_iter < 1:
            raise ValueError("max_iter: not a positive integer")

        if self.init_imp not in InitImpOptions:
            raise ValueError("init_imp: not one of mean, zero, knn")

        if self.parallel not in ParallelOptions:
            raise ValueError("parallel: not one of slurm, local")

        if type(self.n_nodes) != int or self.n_nodes < 1:
            raise ValueError("n_nodes: not a positive integer")
        elif self.n_nodes > p:
            raise ValueError("n_nodes: nodes should be less than variables of dataset")

        if type(self.n_cores) != int or self.n_cores < 1:
            raise ValueError("n_cores: not a positve integer")
        elif self.n_cores > self.n_trees:
            pass
            # raise ValueError("n_cores: cores should be less than n_trees")

        if type(self.n_features) != int or self.n_features < 1:
            raise ValueError("n_features: not a positive integer")
        
        if self.memory is None:
            self.memory = self.n_cores * 2000
        else: 
            pass

    def _init_dirs(self):
        """ private method, initialize hidden files """
        files = ['.out', '.err', '.dat']
        for file in files:
            if os.path.exists(file):
                shutil.rmtree(file)
            os.mkdir(file)

    def fit_transform(self, xmis):
        """ return imputed matrix-like data """
        self._init_dirs()
        self._check_inputs(xmis)

        n_trees = self.n_trees
        max_iter = self.max_iter
        init_imp = self.init_imp
        parallel = self.parallel 
        n_nodes = self.n_nodes 
        n_cores = self.n_cores 
        n_features = self.n_features

        mf = rfimpute.MissForestImputation(n_trees, max_iter, init_imp, parallel, n_nodes, n_cores, n_features)
        res = mf.miss_forest_imputation(xmis)

        return res

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
    mf = MissForest(max_iter=10, init_imp='mean', parallel='slurm', n_nodes=1, n_cores=8, n_features=1024, memory=None)
    start = time()
    ximp = mf.fit_transform(data) 
    duration = time() - start
    rmse = nrmse(nmis, ximp, true_data_)
