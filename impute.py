# Missing Value Imputation Class
# Each feature should be separated for parallel design
from abc import ABC, abstractmethod
from enum import Enum
from randomforest import RandomForest
import abc
import util
import numpy as np
import pickle
import math
import time
import subprocess
import os

class InitialGuessOptions(Enum):
    MEAN    = "mean"
    ZERO    = "zero"
    # KNN     = "knn"
    
class ParallelOptions(Enum):
    SLURM   = "slurm"
    LOCAL   = "local"

class MissForestImputation(ABC):
    """ Private object, do not directly use it """
    def __init__(self, max_iter, init_imp, vart, numi, cati):
        self.max_iter = max_iter
        self.init_imp = init_imp
        self.vart = vart
        self.numi = numi 
        self.cati = cati
        self.vari = None
        self.misi = None
        self.obsi = None
        self.previous_diff        = None
        self.matrix_for_impute    = None
        self.initial_guess_matrix = None
        self.previous_iter_matrix = None
        self.cur_iter_matrix      = None
        self.result_matrix        = None

    @abstractmethod
    def miss_forest_imputation(self, matrix_for_impute):
        pass

    def check_converge(self):
        p = len(self.vart)
        numi = self.numi
        cati = self.cati
        cur_diff = [None, None]
        # difference of numerical
        if len(numi) > 0:
            X_old_num = self.previous_iter_matrix[:, numi]
            X_new_num = self.cur_iter_matrix[:, numi]
            square_diff_sum = np.sum((X_old_num - X_new_num) ** 2)
            square_sum = np.sum((X_new_num) ** 2)
            cur_diff[0] = square_diff_sum / square_sum
        # difference of categorical
        if len(cati) > 0:
            X_old_cat = self.previous_iter_matrix[:, cati]
            X_new_cat = self.cur_iter_matrix[:, cati]
            num_differ = np.sum(X_old_cat != X_new_cat)
            num_mis = sum([self.misi[i] for i in cati])
            cur_diff[1] = num_differ / num_mis

        if self.previous_diff is None:
            self.previous_diff = cur_diff
            return False
        else:
            for i in range(2):
                if self.previous_diff[i] != None and cur_diff[i] > self.previous_diff[i]:
                    return True
            self.previous_diff = cur_diff
            return False

    def raw_fill(self):
        """imputation preparation, fill missing values with specified values"""
        Xmis = self.matrix_for_impute
        Ximp = np.copy(Xmis)
        n, p = np.shape(Xmis)

        misn = [] # number of missing for each variable
        misi = [] # indices of missing samples for each variable
        obsi = [] # indices of observations for each variable
        for v in range(p):
            vt = self.vart[v]
            col = Ximp[:, v]
            var_misi = np.where(np.isnan(col))[0]
            var_obsi = np.delete(np.arange(n), var_misi)
            misn.append(len(var_misi))
            misi.append(var_misi)
            obsi.append(var_obsi)
            if vt == 1: # numerical
                if self.init_imp == InitialGuessOptions.MEAN.value:
                    var_mean = np.mean(col[var_obsi])
                    Ximp[var_misi, v] = np.array([var_mean for _ in range(misn[-1])])
                if self.init_imp == InitialGuessOptions.ZERO.value:
                    Ximp[var_misi, v] = np.array([0 for _ in range(misn[-1])])
            else: # categorical
                if self.init_imp == InitialGuessOptions.MEAN.value:
                    var_mode = util.mode(col[var_obsi].tolist())
                    Ximp[var_misi, v] = np.array([var_mode for _ in range(misn[-1])])
        vari = np.argsort(misn).tolist()
        self.initial_guess_matrix = Ximp
        self.vari = vari
        self.misi = misi
        self.obsi = obsi
