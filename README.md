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
```
MissForest(self, max_iter=10, init_imp='mean', n_estimators=100, 
           max_depth=None, min_samples_split=2, min_samples_leaf=1, 
           min_weight_fraction_leaf=0.0, max_features='sqrt', 
           max_leaf_nodes=None, min_impurity_decrease=0.0, 
           bootstrap=True, random_state=None, verbose=0, 
           warm_start=False, class_weight=None, partition=None, 
           n_cores=1, n_nodes=1, node_features=1, memory=2000, 
           time='1:00:00', parallel='local'):

Parameters
__________
NOTE: Parameters are consisted by MissForest parameters, RandomForest parameters, and SLURM
parameters. For RandomForest is implemented in scikit-learn, many parameters description 
will be directly referred to [2], [3], [4] (who also uses scikit-learn)


```
<!-- ## Contributing -->

## Credits

Coming soon

## Reference

-[1] Stekhoven, Daniel J., and Peter Bühlmann. "MissForest—non-parametric missing value imputation for mixed-type data." Bioinformatics 28.1 (2011): 112-118.
-[2] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
-[3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
-[4] https://github.com/epsilon-machine/missingpy
