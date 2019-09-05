class JobHandler:

    def __init__(self, n_cores, memory, time):
        self.par_quiet = '--quiet'
        self.par_partition = '-p'
        self.par_num_node = '-N'
        self.num_node = '1'
        self.par_num_core_each_node = '-c' 
        self.par_memory = '--mem'
        self.par_time_limit = '--time'
        self.par_job_name = '-J'
        self.job_name = 'impute'
        self.par_output = '-o'
        self.output_ext = '.out'
        self.par_error = '-e'
        self.error_ext = '.err'
        self.script_path = 'job.py'
        self.shell_script_path = '.dat/job.sh'
        self.tmp_X_file = '.dat/tmp_X.dat'

        self.partition = 'production'
        self.num_core_each_node = n_cores
        self.memory = memory
        self.time = time 

    def get_command_shell(self, x_path, argument_path, result_path):
        python_path = 'python'
        script_path = self.script_path
        x_path = x_path
        argument_path = argument_path
        result_path = result_path
        return ([python_path, script_path, x_path, argument_path, result_path])
    
    def get_command(self, node_id, job_id, iter_id):
        exe_path = 'sbatch'
        par_quiet = self.par_quiet
        par_num_node = self.par_num_node
        num_node = self.num_node
        par_num_core_each_node = self.par_num_core_each_node
        num_core_each_node = str(self.num_core_each_node)
        par_memory = self.par_memory
        memory = str(self.memory)
        par_time_limit = self.par_time_limit
        time_limit = self.time 
        par_job_name = self.par_job_name
        job_name = self.job_name  + str(node_id) + '_' + str(job_id) + '_' + str(iter_id)
        par_output = self.par_output
        output_file = '.out/' + job_name + self.output_ext
        par_error = self.par_error
        error_file = '.err/' + job_name + self.error_ext
        shell_script_path = self.shell_script_path

        return ([exe_path, par_quiet, 
                self.par_partition, self.partition, 
                par_num_node, num_node,
                par_num_core_each_node, num_core_each_node, 
                par_memory, memory, par_time_limit, time_limit,
                par_job_name, job_name, par_output, output_file, par_error, 
                error_file, shell_script_path])

    def get_arguments_varidx_file(self, node_id, job_id):
        return '.dat/arguments_' + str(node_id) + '_' + str(job_id) + '.dat'
        
    def get_results_varidx_file(self, node_id, job_id):
        return '.dat/results_' + str(node_id) + '_' + str(job_id) + '.dat'
