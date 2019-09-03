# Missing Value Imputation Class
# Each feature should be separated for parallel design
from enum import Enum
from sklearn.ensemble import RandomForestRegressor

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


class MissForestImputationParameters:
    def __init__(self, max_iter, init_imp, parallel, n_nodes, n_cores, n_features, memory):
        self.max_iter = max_iter
        self.init_imp = init_imp
        self.parallel = parallel
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.slurm_parameters = SlurmImputationParameters(n_cores, memory)
        
        self.tmp_X_file = '.dat/tmp_X.dat'
        
    def get_arguments_varidx_file(self, node_id, job_id):
        s = "-"
        # return '.dat/arguments_varidx_' + s.join([str(x) for x in varidx]) + '.dat'
        return '.dat/arguments_' + str(node_id) + '_' + str(job_id) + '.dat'
        
    def get_results_varidx_file(self, node_id, job_id):
        s = "-"
        # return '.dat/results_varidx_' + s.join([str(x) for x in varidx]) + '.dat'
        return '.dat/results_' + str(node_id) + '_' + str(job_id) + '.dat'
        
class SlurmImputationParameters:
    def __init__(self, n_cores, memory):
        self.par_quiet = '--quiet'
        self.par_num_node = '-N'
        self.num_node = 1
        self.par_num_core_each_node = '-c' 
        self.par_memory = '--mem'
        self.par_time_limit = '--time'
        self.par_job_name = '-J'
        self.job_name = "Imputation"
        self.par_output = '-o'
        self.output_ext = ".out"
        self.par_error = '-e'
        self.error_ext = ".err"
        self.script_path = "job.py"
        self.shell_script_path = 'job.sh'

        self.num_core_each_node = n_cores
        self.memory = memory
        self.time_limit_hr = 0
        self.time_limit_min = 10

    def get_command_shell(self, x_path, argument_path, result_path):
        python_path = 'python'
        exe_path = 'srun'

        script_path = self.script_path
        x_path = x_path
        argument_path = argument_path
        result_path = result_path
        
        return ([
            exe_path, 
            python_path, script_path, x_path, argument_path, result_path])
    
    def get_command(self, node_id, job_id, iter_id):
        exe_path = 'sbatch'
        
        par_quiet = self.par_quiet
        par_num_node = self.par_num_node
        num_node = str(self.num_node)
        par_num_core_each_node = self.par_num_core_each_node
        num_core_each_node = str(self.num_core_each_node)
        par_memory = self.par_memory
        memory = self.memory
        par_time_limit = self.par_time_limit
        # time_limit = str(self.time_limit_hr) + ":" + (format(self.time_limit_min,'02')) + ":00"
        time_limit = "4-00:00"
        par_job_name = self.par_job_name
        s = "-"
        # job_name = self.job_name + '_' + s.join([str(x) for x in varidx])
        job_name = self.job_name  + str(node_id) + '_' + str(job_id) + '_' + str(iter_id)
        par_output = self.par_output
        output_file = '.out/' + job_name + self.output_ext
        par_error = self.par_error
        error_file = '.err/' + job_name + self.error_ext
        
        shell_script_path = self.shell_script_path
        return ([exe_path, par_quiet, par_num_node, num_node, par_num_core_each_node, num_core_each_node, par_memory, memory, par_time_limit, time_limit, \
                par_job_name, job_name, par_output, output_file, par_error, error_file, shell_script_path])

class MissForestImputationArguments_SLURM:
    def __init__(self, rf = None, vari = None, obsi = [], misi = []):
        self.rf = rf
        self.vari = vari
        self.obsi = obsi
        self.misi = misi
        self.results = MissForestImputationResults_SLURM()
        
class MissForestImputationResults_SLURM:
    def __init__(self):
        self.imp_list = []
        self.done = True
        self.err = None #Exception object


class MissForestImputation:
    """ Private object, do not directly use it """
    def __init__(self, n_trees, max_iter, init_imp, parallel, n_nodes, n_cores, n_features, memory):
        self.parameters = MissForestImputationParameters(n_trees, max_iter, init_imp, parallel, n_nodes, n_cores, n_features, memory)
        
        self.matrix_for_impute = None
        self.initial_guess_matrix = None
        self.vari = None
        self.misi = None
        self.obsi = None
        self.previous_iter_matrix = None
        self.previous_diff = None
        self.cur_iter_matrix = None
        self.result_matrix = None
        self.slurm_instance = None

    def miss_forest_imputation(self, matrix_for_impute):
        self.matrix_for_impute = matrix_for_impute
        self.initial_guess() #Prep

        if self.parameters.parallel == ParallelOptions.SLURM.value:
            self.miss_forest_imputation_SLURM()
        elif self.parameters.parallel == ParallelOptions.LOCAL.value:
            self.miss_forest_imputation_local()
        
        return self.result_matrix
        
    def miss_forest_imputation_local(self):
        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1
        
        rf = RandomForestImputation()
        while True:
            if cur_iter > self.parameters.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return
            print("iteration " + str(cur_iter))
            
            for i in range(len(self.vari)):
                cur_X = self.cur_iter_matrix
                _, p = np.shape(cur_X)
                
                cur_vari = self.vari[i]
                cur_obsi = self.obsi[cur_vari]
                cur_misi = self.misi[cur_vari]
                if (len(cur_misi) == 0):
                    continue
                
                p_train = np.delete(np.arange(p), cur_vari)
                X_train = cur_X[cur_obsi, :]
                X_train = X_train[:, p_train]
                
                X_test = cur_X[cur_misi, :]
                X_test = X_test[:, p_train]
                
                y_train = cur_X[cur_obsi, :]
                y_train = y_train[:, cur_vari]
                
                imp = rf.fit_predict(X_train, y_train, X_test)
                print(imp.shape)
                print(self.cur_iter_matrix[cur_misi,cur_vari].shape)
                self.cur_iter_matrix[cur_misi,cur_vari] = imp
                
            #raise Exception('!!!')    
            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return
                
            #Update the previous_iter_matrix
            self.previous_iter_matrix = np.copy(self.cur_iter_matrix)
            
            cur_iter = cur_iter + 1
                
        
    def miss_forest_imputation_SLURM(self):
        vari_node = self.split_var()
        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1
        
        rf = RandomForestImputation()
        
        s = time.time()
        for i in range(len(vari_node)):
            for j in range(len(vari_node[i])):
                cur_vari = vari_node[i][j]
                cur_obsi = []
                cur_misi = []
                for k in range(len(vari_node[i][j])):
                    cur_obsi.append(self.obsi[cur_vari[k]])
                    cur_misi.append(self.misi[cur_vari[k]])
                argument_path = self.parameters.get_arguments_varidx_file(i, j)
                with open(argument_path, 'wb') as tmp:
                    argument_object = MissForestImputationArguments_SLURM(rf, cur_vari, cur_obsi, cur_misi)
                    pickle.dump(argument_object, tmp)
        
        while True:
            if cur_iter > self.parameters.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return
            print("iteration " + str(cur_iter))
            
            for i in range(len(vari_node)):
                s = time.time()
                cur_X = self.cur_iter_matrix
                x_path = self.parameters.tmp_X_file
                with open(x_path, 'wb') as tmp:
                    pickle.dump(cur_X, tmp)
                print("dump X duration:", time.time() - s)
                for j in range(len(vari_node[i])):
                    #Prepare the jobs
                    cur_vari = vari_node[i][j]
                    cur_obsi = []
                    cur_misi = []
                    for k in range(len(vari_node[i][j])):
                        cur_obsi.append(self.obsi[cur_vari[k]])
                        cur_misi.append(self.misi[cur_vari[k]])

                    argument_path = self.parameters.get_arguments_varidx_file(i, j)
                    result_path = self.parameters.get_results_varidx_file(i, j)
                    with open(result_path, 'wb') as tmp:
                        argument_object = MissForestImputationArguments_SLURM(rf, cur_vari, cur_obsi, cur_misi)
                        argument_object.results.done = False
                        pickle.dump(argument_object.results, tmp)
                    
                    #Submit the jobs
                    #Write the bash
                    command_shell = self.parameters.slurm_parameters.get_command_shell(x_path, argument_path, result_path)
                    command_shell =' '.join(command_shell)
                    with open(self.parameters.slurm_parameters.shell_script_path,'w') as tmp:
                        tmp.writelines('#!/bin/bash\n')
                        tmp.writelines(command_shell)
                    
                    command = self.parameters.slurm_parameters.get_command(i, j, cur_iter)
                    subprocess.call(command)
                
                # print('Polling!')
                #Polling:
                finish = False
                finished_ind = [False]*len(vari_node[i])
                finished_count = 0
                while finish == False:
                    time.sleep(0.1)
                    finish = True
                    for j in range(len(vari_node[i])):
                        if finished_ind[j] == True:
                            continue
                            
                        cur_vari = vari_node[i][j]
                        cur_obsi = []
                        cur_misi = []
                        for k in range(len(vari_node[i][j])):
                            cur_obsi.append(self.obsi[cur_vari[k]])
                            cur_misi.append(self.misi[cur_vari[k]])
                            
                        result_path = self.parameters.get_results_varidx_file(i, j)
                        try:
                            with open(result_path,'rb') as tmp:
                                cur_result = pickle.load(tmp)
                                if cur_result.done == False:
                                    finish = False
                                    break
                                else:
                                    for k in range(len(cur_vari)):
                                        self.cur_iter_matrix[cur_misi[k],cur_vari[k]] = cur_result.imp_list[k]
                                    finished_ind[j] = True

                            if finished_ind.count(True) > finished_count:
                                finished_count = finished_ind.count(True)
                                print(finished_count, "/", len(finished_ind), "finished!")
                                
                        except Exception as e:
                            finish = False
                            break

            #raise Exception('!!!')    
            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return
                
            #Update the previous_iter_matrix
            self.previous_iter_matrix = np.copy(self.cur_iter_matrix)
            
            cur_iter = cur_iter + 1
        
    def split_var(self):
        #[NODES,[JOBS,[FEATURE]],]
    
        vari_node = []
        cur_node_idx = 0
        cur_job_idx = 0
        
        cur_jobs = []
        cur_vari = []
        
        for var in vari:
            cur_vari.append(var)
            if len(cur_vari) == self.parameters.n_features:
                cur_jobs.append(cur_vari)
                cur_vari = []
                if len(cur_jobs) == self.parameters.n_nodes:
                    vari_node.append(cur_jobs)
                    cur_jobs = []
        
        if len(cur_vari) > 0:
            cur_jobs.append(cur_vari)
        if len(cur_jobs) > 0:
            vari_node.append(cur_jobs)
            
        print(np.shape(vari_node))
        return vari_node

    def check_converge(self):
        diff_A = 0
        diff_B = 0

        diff_A = np.sum((self.previous_iter_matrix - self.cur_iter_matrix)**2)
        diff_B = np.sum((self.cur_iter_matrix)**2)
        
        # print(self.previous_iter_matrix)
        # print(self.cur_iter_matrix)
        
        # print(diff_A)
        # print(diff_B)

        cur_diff = diff_A/diff_B
        # print(self.previous_diff)
        # print(cur_diff)
        # print('')
        if self.previous_diff is None:
            self.previous_diff = cur_diff
            return False
        else:
            if cur_diff > self.previous_diff:
                return True
            else:
                self.previous_diff = cur_diff
                return False
            # else:
                # return False
                
    def initial_guess(self):
        if self.parameters.init_imp == InitialGuessOptions.MEAN.value:
            self.initial_guess_average()
            
    def initial_guess_average(self, nan=None):
        Xmis = self.matrix_for_impute
    
        # Input
        #   Xmis: missing-valued matrix
        #     nan : string indicating NaN in the given Xmis, defualt as float("nan")
        # Output
        #     Ximp: raw-imputed matrix
        #     vari: list of indices sorted by the number of missing values in 
        #           ascending order
        #     misi: list of indices of missing values for each variable
        #      obsi: list of indices of observed values for each variable
        try:
            n, p = np.shape(Xmis)
        except:
            raise ValueError("Xmis is not a matrix")
        
        if nan is not None and type(nan) is not str:
            raise ValueError("nan is either None or a string")

        # start initial imputation
        Ximp = np.copy(Xmis)

        misn = [] # number of missing for each variable
        misi = [] # indices of missing samples for each variable
        obsi = [] # indices of observations for each variable
        for v in range(p):
            cnt = 0
            col = Ximp[:, v]
            var_misi, var_obsi = [], []
            for i in range(n):
                if nan is None:
                    if math.isnan(col[i]):
                        var_misi.append(i)
                        cnt += 1
                    else:
                        var_obsi.append(i)
                else:
                    if col[i] == nan:
                        var_misi.append(i)
                        cnt += 1
                    else:
                        var_obsi.append(i)
            
            misn.append(cnt)
            var_obs = col[var_obsi]
            
            var_mean = np.mean(var_obs)

            for i in range(len(var_misi)):
                Ximp[var_misi[i], v] = var_mean
            
            misi.append(var_misi)
            obsi.append(var_obsi)
        vari = np.argsort(misn).tolist()

        self.initial_guess_matrix = Ximp
        self.vari = vari
        self.misi = misi
        self.obsi = obsi
        

        

class RandomForestImputation(object):

    def __init__(self, n_estimators=100,
         criterion="mse", max_depth=None, min_samples_split=2,
         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="sqrt",
         max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
         oob_score=False, n_jobs=-1, random_state=None, verbose=0,
         warm_start=False):
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
            regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, max_features='sqrt')
            regr.fit(X_train, y)
            imp = regr.predict(X_test)
            self.done = True
        except Exception as e:
            self.err = e
        
        if imp is None:
            raise Exception('NONE!')

        return imp
