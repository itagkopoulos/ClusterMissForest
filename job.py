#!/usr/bin/env python
# job.py
# Currently only working for numerical values

from sklearn.ensemble import RandomForestRegressor
import sys
import pickle
import numpy as np 

class RandomForestImputation(object):

    def __init__(self, n_estimators=100,
         criterion="mse", max_depth=None, min_samples_split=2,
         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="sqrt",
         max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
         oob_score=False, n_jobs=-1, random_state=None, verbose=0,
         warm_start=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features 
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score 
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.err = None
        self.done = False

    def _check_input(self):
        pass

    def fit_predict(self, X_train, y, X_test):
        imp = None
        try:
            '''regr = RandomForestRegressor(
            self.n_estimators,
            self.criterion,
            self.max_depth, 
            self.min_samples_split, 
            self.min_samples_leaf, 
            self.min_weight_fraction_leaf, 
            self.max_features,  
            self.max_leaf_nodes,  
            self.min_impurity_decrease, 
            self.bootstrap, 
            self.oob_score,  
            self.n_jobs, 
            self.random_state, 
            self.verbose, 
            self.warm_start)'''
            regr = RandomForestRegressor(n_estimators = 100, verbose=1, n_jobs = -1)
            regr.fit(X_train, y)
            imp = regr.predict(X_test)
            self.done = True
        except Exception as e:
            self.err = e
        
        if imp is None:
            raise Exception('NONE!')

        return imp

if __name__ == "__main__":

    data_file = sys.argv[1]
    res_file = sys.argv[2]
    
    with open(data_file, "rb") as tmp:
        X = pickle.load(tmp)
    with open(res_file, "rb") as tmp:
        res_obj = pickle.load(tmp)

    vari = res_obj.vari
    misi = res_obj.misi
    obsi = res_obj.obsi
    # TODO
    #      sklearn parameter files

    X_array = np.array(X)
    _, p = np.shape(X_array)

    p_train = np.delete(np.arange(p), vari)
    print(p_train)
    
    X_train = X_array[obsi, :]
    X_train = X_train[:, p_train]
    
    X_test = X_array[misi, :]
    X_test = X_test[:, p_train]
    
    y_train = X_array[obsi, :]
    y_train = y_train[:, vari]

    rf = RandomForestImputation()
    imp = rf.fit_predict(X_train, y_train, X_test)

    res_obj.imp = imp
    res_obj.done = rf.done
    res_obj.err = rf.err
    
    
    with open(res_file, "wb") as tmp:
        pickle.dump(res_obj, tmp)





























