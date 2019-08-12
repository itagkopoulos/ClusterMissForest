# preprecess.py

from collections import defaultdict
import math
import numpy as np 

def raw_transform(xmis, nan=None):
	# Input
	#   xmis: missing-valued matrix
	# 	nan : string indicating NaN in the given xmis, defualt as float("nan")
	# Output
	# 	xobs: raw-imputed matrix
	# 	vari: list of indices sorted by the number of missing values in 
	# 		  ascending order
	# 	misi: list of indices of missing values for each variable
	#  	obsi: list of indices of observed values for each variable
	try:
		n, p = np.shape(xmis)
	except:
		raise ValueError("X is not a matrix")
	
	if nan is not None and type(nan) is not str:
		raise ValueError("nan is either None or a string")

	# start initial imputation
	xobs = copy(xmis)

	misn = [] # number of missing for each variable
	misi = [] # indices of missing samples for each variable
	obsi = [] # indices of observations for each variable
	for v in range(p):
		col = [l[v] for l in xobs]
		var_misi, var_obsi = [], []
		for i in range(n):
			if nan is None:
				if math.isnan(col[i]):
					var_misi.append(i)
				else:
					var_obsi.append(i)
			else:
				if col[i] == nan:
					var_misi.append(i)
				else:
					var_obsi.append(i)
		var_obs = [col[j] for j in var_obsi]
		# numerical variable
		if isinstance(X[0][v], (float, np.floating))
			var_mean = np.mean(var_obs)
			for i in var_misi:
				xobs[i][v] = var_mean
		# categorical variable
		else:
			classes = defaultdict(int)
			for obs in var_obs:
				if not np.isnan(obs):
					classes[obs] += 1
			var_max = max(classes, key=classes.get)
			for i in var_misi:
				xobs[i][v] = var_max  
		misi.append(var_misi)
		obsi.append(var_obsi)
	vari = np.argsort(misn).tolist()
	
	return xobs, vari, misi, obsi
