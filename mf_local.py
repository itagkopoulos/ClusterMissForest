from impute import MissForestImputation
from randomforest import RandomForest
import numpy as np 

class MissForestImputationLocal(MissForestImputation):

    def __init__(self, mf_params, rf_params):
        super().__init__(**mf_params)
        self.params = rf_params

    def miss_forest_imputation(self, matrix_for_impute):
        self.matrix_for_impute = matrix_for_impute
        self.raw_fill()

        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1
        
        while True:
            if cur_iter > self.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return
            print("Iteration " + str(cur_iter))

            for var in self.vari:
                p = len(self.vart)
                vt = self.vart[var]
                cur_X = self.cur_iter_matrix
                cur_obsi = self.obsi[var]
                cur_misi = self.misi[var]
                if (len(cur_misi) == 0):
                    continue
                p_train = np.delete(np.arange(p), var)
                X_train = cur_X[cur_obsi, :][:, p_train]
                y_train = cur_X[cur_obsi, :][:, var]
                X_test = cur_X[cur_misi, :][:, p_train]
                rf = RandomForest(self.params)
                imp = rf.fit_predict(X_train, y_train, X_test, vt)
                self.cur_iter_matrix[cur_misi, var] = imp

            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return
            else:
                self.previous_iter_matrix = np.copy(self.cur_iter_matrix)
                cur_iter = cur_iter + 1
