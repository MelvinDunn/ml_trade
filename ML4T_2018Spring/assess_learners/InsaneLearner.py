"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""
import numpy as np
import LinRegLearner as lrl

class InsaneLearner(object):

    def __init__(self, verbose = False):
        pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'mdunn34' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        self.learners = []
        for i in range(0,20):
            learner = lrl.LinRegLearner()
            n_prime = np.array(np.random.randint(dataY.shape[0], size=int(dataY.shape[0] * .6)))      
            learner.addEvidence(dataX[n_prime, :], dataY[n_prime])
            self.learners.append(learner)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        results = np.zeros(points.shape[0]).astype(float)
        i = 0
        for model in self.learners:
            results += np.array(model.query(points)).astype(float)
            i += 1.
        return results / float(i)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
