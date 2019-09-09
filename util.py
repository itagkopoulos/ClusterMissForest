from collections import defaultdict
import csv
import numpy as np 

def v_parse(val):
    """parse a string to float (continuous) or string (discrete)"""
    try:
        # check if value is int (discrete)
        val = str(int(val))
    except:
    	try:
    		val = float(val)
    	except:
    		val = val 

    return val 

def read_csv(file, header=False):
    """return a parsed csv file in built-in dataframe"""
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if header:
                continue
            line = [v_parse(val) for val in row]
            data.append(line)

    return data 

def isnan(x):
    print(x)
    if isinstance(x, np.floating):
        return np.isnan(x)
    else:
        return x == 'nan'

def mode(x):
	count = defaultdict(int)
	for item in x:
		count[item] += 1
	return max(count, key=count.get)
