from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class RandomForest:

    def __init__(self, n_estimators=100, criterion="mse", max_depth=None,
         min_samples_split=2, min_samples_leaf=1,
         min_weight_fraction_leaf=0.0, max_features="sqrt",
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
            regr = RandomForestRegressor(n_estimators=100, verbose=0, n_jobs=-1, max_features='sqrt')
            regr.fit(X_train, y)
            imp = regr.predict(X_test)
            self.done = True
        except Exception as e:
            self.err = e
        
        if imp is None:
            raise Exception('NONE!')

        return imp
