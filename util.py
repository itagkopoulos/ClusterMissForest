from collections import defaultdict
import csv
import numpy as np 

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
    """return boolean for checking if value is nan"""
    if isinstance(x, (float, np.floating)):
        return np.isnan(x)
    else:
        return x == 'nan'

def mode(x):
    """return the mode of a list"""
	count = defaultdict(int)
	for item in x:
		count[item] += 1
	return max(count, key=count.get)
