from randomforest import RandomForest
from impute import MissForestImputation
from job_handler import JobHandler
import subprocess 
import pickle
import time
import numpy as np

class MissForestImputationSlurmArgumentObject:
    """private class, contains parameters needed by jobs

    parameters
    ----------
    rf_job  : object contains random forest model
    vart    : list of variable types
    vari    : list of variable indices in order of ascending number of 
              missing values
    obsi    : list of lists of indices of observed values for each variable
    misi    : list of lists of ondices of missing values for each variable
    results : object stores the result of finished job on cluster"""
    def __init__(self, rf_obj, vart, vari, obsi, misi):
        self.rf_obj = rf_obj
        self.vart = vart
        self.vari = vari
        self.obsi = obsi
        self.misi = misi
        self.results = MissForestImputationSlurmResultObject()
        
class MissForestImputationSlurmResultObject:
    """private class, stores the result of finished job on cluster

    parameters
    ----------
    imp_list : list records the batch of variables sent to the node
    done     : boolean indicates the job has been done
    err      : error information if job failed
    time     : duration of a job"""
    def __init__(self):
        self.imp_list = []
        self.done = False 
        self.err = None 
        self.time = None 

class MissForestImputationSlurm(MissForestImputation):

    def __init__(self, mf_params, rf_params, partition, n_nodes, n_cores, node_features, memory, time):
        super().__init__(**mf_params)
        self.params = rf_params
        self.n_nodes = n_nodes
        self.node_features = node_features
        self.handler = JobHandler(partition, n_cores, memory, time)

    def miss_forest_imputation(self, matrix_for_impute):
        self.matrix_for_impute = matrix_for_impute
        self.raw_fill()

        vari_node = self.split_var()
        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1
        
        while True:
            if cur_iter > self.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return
            print("iteration " + str(cur_iter))
            
            for i in range(len(vari_node)):
                cur_X = self.cur_iter_matrix
                x_path = self.handler.tmp_X_file
                with open(x_path, 'wb') as tmp:
                    pickle.dump(cur_X, tmp)
                for j in range(len(vari_node[i])):
                    #Prepare the jobs
                    cur_vari = vari_node[i][j]
                    cur_vart = []
                    cur_obsi = []
                    cur_misi = []
                    for k in range(len(vari_node[i][j])):
                        cur_vart.append(self.vart[cur_vari[k]])
                        cur_obsi.append(self.obsi[cur_vari[k]])
                        cur_misi.append(self.misi[cur_vari[k]])

                    argument_path = self.handler.get_arguments_varidx_file(i, j)
                    result_path = self.handler.get_results_varidx_file(i, j)
                    rf = RandomForest(self.params)
                    with open(argument_path, 'wb') as tmp:
                        argument_object = MissForestImputationSlurmArgumentObject(rf, cur_vart, cur_vari, cur_obsi, cur_misi)
                        pickle.dump(argument_object, tmp)
                    with open(result_path, 'wb') as tmp:
                        # argument_object.results.done = False
                        pickle.dump(argument_object.results, tmp)
                    
                    # write job.sh and submit
                    command_shell = self.handler.get_command_shell(x_path, argument_path, result_path)
                    command_shell =' '.join(command_shell)
                    with open(self.handler.shell_script_path, 'w') as tmp:
                        tmp.writelines('#!/bin/bash\n')
                        tmp.writelines(command_shell)
                    command = self.handler.get_command(i, j, cur_iter)
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
                            
                        result_path = self.handler.get_results_varidx_file(i, j)
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
        
        for var in self.vari:
            cur_vari.append(var)
            if len(cur_vari) == self.node_features:
                cur_jobs.append(cur_vari)
                cur_vari = []
                if len(cur_jobs) == self.n_nodes:
                    vari_node.append(cur_jobs)
                    cur_jobs = []
        
        if len(cur_vari) > 0:
            cur_jobs.append(cur_vari)
        if len(cur_jobs) > 0:
            vari_node.append(cur_jobs)
            
        print(np.shape(vari_node))
        return vari_node

