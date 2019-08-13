#Missing Value Imputation Class
#Each feature should be separated for parallel design
from enum import Enum


import numpy as np
import pickle
import math
import time

import subprocess
import copy

class InitialGuessOptions(Enum):
    AVERAGE = "average"
    ZERO    = "zero"
    KNN     = "knn"
    
class ParallelOptions(Enum):
    SLURM   = "slurm"
    LOCAL   = "local"


class MissForestImputationParameters:
    def __init__(self):
        self.initial_guess_mode = InitialGuessOptions.AVERAGE.value
        self.parallel_options = ParallelOptions.SLURM.value
        self.max_iter = 10
        self.num_node = 4
        self.num_core_local = 12
        self.slurm_parameters = SlurmImputationParameters()
        
        self.tmp_X_file = 'tmp_X.dat'
        
    def get_results_varidx_file(self, varidx):
        return 'results_varidx_' + str(varidx) + '.dat'
        
class SlurmImputationParameters:
    def __init__(self):
        self.par_num_node = '-N'
        self.num_node = 1
        self.par_num_core_each_node = '-c'
        self.num_core_each_node = 2
        self.par_time_limit = '--time'
        self.time_limit_hr = 1
        self.time_limit_min = 0
        self.par_job_name = '-J'
        self.job_name = "Imputation"
        self.par_output = '-o'
        self.output_ext = ".output"
        self.par_error = '-e'
        self.error_ext = ".error"
        
        self.script_path = "job.py"
        self.shell_script_path = 'job.sh'
        
    def get_command_shell(self, x_path, result_path):
        python_path = 'python'
        exe_path = 'srun'

        script_path = self.script_path
        x_path = x_path
        result_path = result_path
        
        return ([exe_path, python_path, script_path, x_path, result_path])
    
    def get_command(self, varidx):
        exe_path = 'sbatch'
        
        par_num_node = self.par_num_node
        num_node = str(self.num_node)
        par_num_core_each_node = self.par_num_core_each_node
        num_core_each_node = str(self.num_core_each_node)
        par_time_limit = self.par_time_limit
        time_limit = str(self.time_limit_hr) + ":" + (format(self.time_limit_min,'02')) + ":00"
        par_job_name = self.par_job_name
        job_name = self.job_name + '_' + str(varidx)
        par_output = self.par_output
        output_file = job_name + self.output_ext
        par_error = self.par_error
        error_file = job_name + self.error_ext
        
        shell_script_path = self.shell_script_path
        return ([exe_path, par_num_node, num_node, par_num_core_each_node, num_core_each_node, par_time_limit, time_limit, \
                par_job_name, job_name, par_output, output_file, par_error, error_file, shell_script_path])

class MissForestImputationResults_SLURM:
    def __init__(self, vari = None, obsi = [], misi = []):
        self.imp = None
        self.vari = vari
        self.obsi = obsi
        self.misi = misi
        self.done = False
        self.err = None #Exception object
        
        
        

class MissForestImputation:
    def __init__(self):
        self.parameters = MissForestImputationParameters()
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

        if self.parameters.parallel_options == ParallelOptions.SLURM.value:
            self.miss_forest_imputation_SLURM()
            
        
        return self.result_matrix
        
    def miss_forest_imputation_SLURM(self):
        vari_node = self.split_var()
        self.previous_iter_matrix = copy.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = copy.copy(self.initial_guess_matrix)
        cur_iter = 0
        while True:
            if cur_iter >= self.parameters.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return
        
            for i in range(len(vari_node)):
                cur_X = self.cur_iter_matrix
                
                x_path = self.parameters.tmp_X_file
                
                with open(x_path, 'wb') as tmp:
                    pickle.dump(cur_X, tmp)
                    
                for j in range(len(vari_node[i])):
                    #Prepare the jobs
                    cur_vari = vari_node[i][j]
                    cur_obsi = self.obsi[cur_vari]
                    cur_misi = self.misi[cur_vari]
                    
                    result_path = self.parameters.get_results_varidx_file(cur_vari)
                    with open(result_path, 'wb') as tmp:
                        results_object = MissForestImputationResults_SLURM(cur_vari, cur_obsi, cur_misi)
                        pickle.dump(results_object, tmp)
                        
                    #Submit the jobs
                    #Write the bash
                    command_shell = self.parameters.slurm_parameters.get_command_shell(x_path, result_path)
                    command_shell =' '.join(command_shell)
                    with open(self.parameters.slurm_parameters.shell_script_path,'w') as tmp:
                        tmp.writelines('#!/bin/bash\n')
                        tmp.writelines(command_shell)
                    
                    command = self.parameters.slurm_parameters.get_command(cur_vari)
                    subprocess.call(command)
                
                
                print('Polling!')
                #Polling:
                finish = False
                while finish == False:
                    time.sleep(1)
                    finish = True
                    
                    for j in range(len(vari_node[i])):
                        cur_vari = vari_node[i][j]
                        result_path = self.parameters.get_results_varidx_file(cur_vari)
                        try:
                            with open(result_path,"rb") as tmp:
                                cur_result = pickle.load(tmp)

                        except Exception as e:
                            finish = False
                            break
                            
                        if cur_result.done == False:
                            finish = False
                            break
                            
                            
                
                #Update the cur_iter_matrix
                for j in range(len(vari_node[i])):
                    cur_vari = vari_node[i][j]
                    result_path = self.parameters.get_results_varidx_file(cur_vari)
                    cur_result = pickle.load(open(result_path,"rb"))
                    cur_misi = cur_result.misi
                    cur_obsi = cur_result.obsi
                    
                    '''from sklearn.ensemble import RandomForestRegressor
                    regr = RandomForestRegressor(n_estimators = 100)
                    p_train = np.delete(np.arange(len(self.vari)), cur_vari)
                    tmp_X = cur_X[cur_obsi,:]
                    tmp_X = tmp_X[:,p_train]
                    
                    regr.fit(tmp_X, cur_X[cur_obsi,cur_vari])
                    
                    tmp_X = cur_X[cur_misi,:]
                    tmp_X = tmp_X[:,p_train]
                    
                    imp = regr.predict(tmp_X)
                    
                    print(imp)
                    print(cur_result.imp)'''
                    
                    
                    self.cur_iter_matrix[cur_misi,cur_result.vari] = cur_result.imp
                    
                    
            #raise Exception('!!!')    
            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return
                
            #Update the previous_iter_matrix
            self.previous_iter_matrix = copy.copy(self.cur_iter_matrix)
            
            cur_iter = cur_iter + 1
        
    def split_var(self):
        vari_node = []
        cur_idx = 0
        cur_vari = []
        
        for i in range(len(self.vari)):
            
            if cur_idx == self.parameters.num_node:
                vari_node.append(cur_vari)
                cur_vari = [self.vari[i]]
                cur_idx = 0
                if i == (len(self.vari)-1):
                    vari_node.append(cur_vari)
            else:
                cur_vari.append(self.vari[i])
                if i == (len(self.vari)-1):
                    vari_node.append(cur_vari)
                
            cur_idx = cur_idx + 1
        print(vari_node) 
        return vari_node

    def check_converge(self):
        diff_A = 0
        diff_B = 0
        '''
        for i in range(len(self.vari)):
            cur_vari = self.vari[i]
            result_path = self.parameters.get_results_varidx_file(cur_vari)
            cur_result = pickle.load(open(result_path,"rb"))
            cur_misi = cur_result.misi

            old_val = self.previous_iter_matrix[cur_misi,cur_result.vari]
            new_val = self.cur_iter_matrix[cur_misi,cur_result.vari]
            
            if cur_vari == 1:
                print('==============')
                print(cur_misi)
                print(cur_vari)
                print(old_val)
                print(new_val)
                print('===============')
            
            
            diff_A += np.sum((old_val-new_val)**2)
            diff_B += np.sum(new_val**2)
            
            print((old_val-new_val)**2)
            print(new_val**2)
        '''
        diff_A = np.sum((self.previous_iter_matrix - self.cur_iter_matrix)**2)
        diff_B = np.sum((self.cur_iter_matrix)**2)
        
        print(self.previous_iter_matrix)
        print(self.cur_iter_matrix)
        
        print(diff_A)
        print(diff_B)

        cur_diff = diff_A/diff_B
        print(self.previous_diff)
        print(cur_diff)
        print('')
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
        if self.parameters.initial_guess_mode == InitialGuessOptions.AVERAGE.value:
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
        
        
