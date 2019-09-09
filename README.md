# ClusterMissForest

missForest implemented in Python with the support to high-performance computing. 
A fast approach of parallelizing missing value imputation task on cluster
computers. In order to fully utilize the advantage provided by HPC, this package approaches the missing value
imputation task by parallelizing in two different steps. First, the variables of dataset will be splitted and 
imputed in parallel on different nodes (machines), and, second, the estimators will be spliited and computed
on different cores. The larger datasets are, the less influence caused by overhead of parallelization.

## Installation

Coming soon

## Usage

Coming soon
<!-- ## Contributing -->

## Credits

Coming soon