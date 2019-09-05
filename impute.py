# Missing Value Imputation Class
# Each feature should be separated for parallel design
from abc import ABC, abstractmethod
from enum import Enum
from randomforest import RandomForest
import abc
import numpy as np
import pickle
import math
import time
import subprocess
import os

class InitialGuessOptions(Enum):
    MEAN    = "mean"
    ZERO    = "zero"
    KNN     = "knn"
    
class ParallelOptions(Enum):
    SLURM   = "slurm"
    LOCAL   = "local"

class MissForestImputation(ABC):
    """ Private object, do not directly use it """
    def __init__(self, max_iter, init_imp, vart):
        self.max_iter = max_iter
        self.init_imp = init_imp
        self.vart = vart
        self.initial_guess_matrix = None
        self.vari = None
        self.misi = None
        self.obsi = None
        self.previous_iter_matrix = None
        self.cur_iter_matrix = None
        self.result_matrix = None
        self.previous_diff = None
        self.matrix_for_impute = None

    @abstractmethod
    def miss_forest_imputation(self, matrix_for_impute):
        pass

    def check_converge(self):
        diff_A = np.sum((self.previous_iter_matrix - self.cur_iter_matrix)**2)
        diff_B = np.sum((self.cur_iter_matrix)**2)
        cur_diff = diff_A / diff_B
        if self.previous_diff is None:
            self.previous_diff = cur_diff
            return False
        else:
            if cur_diff > self.previous_diff:
                return True
            else:
                self.previous_diff = cur_diff
                return False
                
    def initial_guess(self):

        if self.init_imp == InitialGuessOptions.MEAN.value:
            self.initial_guess_average()
            
    def initial_guess_average(self):
        Xmis = self.matrix_for_impute
        Ximp = np.copy(Xmis)
        n, p = np.shape(Xmis)

        misn = [] # number of missing for each variable
        misi = [] # indices of missing samples for each variable
        obsi = [] # indices of observations for each variable
        for v in range(p):
            col = Ximp[:, v]
            var_misi = np.where(np.isnan(col))[0]
            var_obsi = np.delete(np.arange(n), var_misi)
            var_mean = np.mean(col[var_obsi])
            misn.append(len(var_misi))
            Ximp[var_misi, v] = np.array([var_mean for _ in range(misn[-1])])
            misi.append(var_misi)
            obsi.append(var_obsi)
        vari = np.argsort(misn).tolist()
        self.initial_guess_matrix = Ximp
        self.vari = vari
        self.misi = misi
        self.obsi = obsi
