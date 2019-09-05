#!/usr/bin/env python
# job.py
# Currently only working for numerical values
from time import time

import sys
import pickle
import numpy as np 

import rfimpute
import mmap


if __name__ == "__main__":

    print("Start job ...")
    
    data_file = sys.argv[1]
    arg_file = sys.argv[2]
    res_file = sys.argv[3]
    
    start = time()
    with open(data_file, "r+b") as tmp:
        mm = mmap.mmap(tmp.fileno(), 0)
        X = pickle.load(mm)
    with open(arg_file, "rb") as tmp:
        arg_obj = pickle.load(tmp)
    duration_load = time() - start

    X_array = np.array(X)
    imp_list = []
    
    print("Duration of load:", duration_load)
    for i in range(len(arg_obj.vari)):
        vart = arg_obj.vart
        vari = arg_obj.vari[i]
        misi = arg_obj.misi[i]
        obsi = arg_obj.obsi[i]
        # TODO
        #      sklearn parameter files

        print("Start variable", i, '...')
        _, p = np.shape(X_array)

        p_train = np.delete(np.arange(p), vari)
        X_train = X_array[obsi, :]
        X_train = X_train[:, p_train]
        X_test = X_array[misi, :]
        X_test = X_test[:, p_train]
        y_train = X_array[obsi, :]
        y_train = y_train[:, vari]

        start = time()
        rf = arg_obj.rf_obj
        imp = rf.fit_predict(X_train, y_train, X_test, vart[i])
        duration_rf = time() - start
        
        X_array[misi,vari] = imp
        imp_list.append(imp)

        print("Duration of rf  :", duration_rf)

    arg_obj.results.done = True
    arg_obj.results.imp_list = imp_list
    
    start = time()
    with open(res_file, "wb") as tmp:
        pickle.dump(arg_obj.results, tmp)
    duration_dump = time() - start

    print("Duration of dump:", duration_dump)





























