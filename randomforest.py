from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class RandomForest:

    # def __init__(self, n_estimators, criterion, max_depth,
    #      min_samples_split, min_samples_leaf,
    #      min_weight_fraction_leaf, max_features,
    #      max_leaf_nodes, min_impurity_decrease, bootstrap,
    #      oob_score, n_jobs, random_state, verbose,
    #      warm_start, class_weight):
    #     self.n_estimators = n_estimators
    #     self.criterion = criterion
    #     self.max_depth = max_depth
    #     self.min_samples_split = min_samples_split
    #     self.min_samples_leaf = min_samples_leaf
    #     self.min_weight_fraction_leaf = min_weight_fraction_leaf
    #     self.max_features = max_features 
    #     self.max_leaf_nodes = max_leaf_nodes 
    #     self.min_impurity_decrease = min_impurity_decrease
    #     self.bootstrap = bootstrap
    #     self.oob_score = oob_score 
    #     self.n_jobs = n_jobs
    #     self.random_state = random_state
    #     self.verbose = verbose
    #     self.warm_start = warm_start
    #     self.err = None
    #     self.done = False
    def __init__(self, params, cw):
        """ integrated random forest model """
        self.reg = RandomForestRegressor(criterion='mse', **params)
        self.clf = RandomForestClassifier(criterion='gini', **params, class_weight=cw)
        
    def fit_predict(self, X_train, y, X_test, vt):
        imp = None
        try:
            rf = None
            if vt == 1:
                rf = self.reg 
            else:
                rf = self.clf 
            rf.fit(X_train, y)
            imp = rf.predict(X_test)
            self.done = True
        except Exception as e:
            self.err = e

        return imp
