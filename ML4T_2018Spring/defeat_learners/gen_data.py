"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    # linear regression assumes gaussian distribution.
    X = (np.random.randn(1000,30))
    Y = np.random.random(size = (1000,))*200-100
    # Here's is an example of creating a Y from randomly generated
    # X with multiple columns
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    # this will represent multicolinerity to linear regression.
    # also your number of columns, should be much much smaller than your numbe of rows.
    X = (np.random.rand(300,300))
    Y = np.random.random(size = (300,))*200-100
    return X, Y

def author():
    return 'mdunn34' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
