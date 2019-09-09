# ClusterMissForest

missForest implemented in Python with the support to high-performance computing. 
A fast approach of parallelizing missing value imputation task on cluster
computers. In order to fully utilize the advantage provided by HPC, this package approaches the missing value
imputation task by parallelizing in two different steps. First, the variables of dataset will be splitted and 
imputed in parallel on different nodes (machines), and, second, within each node, the estimators will be spliited 
and computed on different cores.

ClusterMissForest is relied on RandomForestRegressor and RandomForestClassifier of scikit-learn, and, therefore, 
it is currently not available to directly take categorical variables. Instead, please use one-hot encoder to 
transform your dataset. We are working on adding this feature.

## Installation

Coming soon

## Example
- If you run on 'local' mode, _n_cores_ is the same as _n_jobs_ of scikit-learn. 
```python
>>> import numpy as np
>>> Xmis = np.array([[1.0, 2.0, np.nan], [1.1, 2.2, 3.3], [1.5, np.nan, 5.0]])
>>> mf = MissForest(max_iter=10, n_estimators=100, n_cores=8, parallel='local')
>>> mf.fit_transform(Xmis)
Iteration 1
Iteration 2
array([[1.   , 2.   , 4.15 ],
       [1.1  , 2.2  , 3.3  ],
       [1.5  , 2.098, 5.   ]])
```
- If you run on 'slurm' mode, make sure you have accessed in machines that have installed SLURM.
```python
>>> mf = MissForest(max_iter=10, n_estimators=100, n_nodes=2, n_cores=8, parallel='slurm')
>>> mf.fit_transform(Xmis)
(2,)
iteration 1
Submitted batch job 4836926
Submitted batch job 4836927
Submitted batch job 4836928
iteration 2
Submitted batch job 4836929
Submitted batch job 4836930
Submitted batch job 4836931
iteration 3
Submitted batch job 4836932
Submitted batch job 4836933
Submitted batch job 4836934
array([[1.   , 2.   , 4.116],
       [1.1  , 2.2  , 3.3  ],
       [1.5  , 2.112, 5.   ]])
```

## API

<!-- ## Contributing -->

## Credits

Coming soon

## Reference

Coming soon
