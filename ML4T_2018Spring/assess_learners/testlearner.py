"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import sys
import pandas as pd
from timeit import default_timer as timer


if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    print(inf)
    if 'data/Istanbul.csv' in sys.argv[1]:
        print("Istanbul")
        data = np.array(pd.read_csv(inf))
        data = data[1:,1:]
    else:
        data = np.array(pd.read_csv(inf))
    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]


    # create a learner and train it
    #learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":4}, bags = 20, boost = False, verbose = False)
    #learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
    #learner = bl.BagLearner(learner = it.InsaneLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
    #learner = dt.DTLearner(1)

    start = timer()
    learner = dt.DTLearner(2)
    learner.addEvidence(trainX, trainY) # train it
    end = timer()
    print("time",end - start)
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=(trainY).astype(float))
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef((predY), y=(testY).astype(float))
    print "corr: ", c[0,1]
