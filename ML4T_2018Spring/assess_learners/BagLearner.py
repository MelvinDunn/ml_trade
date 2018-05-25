"""
Given a standard training set D of size n, bagging
generates new training sets Di, each of size n',
by sampling from D uniformly and with replacement.

By sampling with replacement, some observations
may be repeated in each Di. if n'=n, then for large n
the set Di is expected to have a fraction (1-1/e)
of the unique exmaples of D, the rest being
duplicates. This kind of sample is known as
a bootstrap sample. The m models are fitted using
the above m bootstrap samples and combined
by averaging the output (for regression) or voting
(for classification)

Bagging leads to improvements for unstable
procedures (Breiman, 1996 which include),
for example, artificial neural networks,
classification and regression trees,
and subset selection
"""

import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        

    def author(self):
        return 'mdunn34' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.learners = []
        for i in range(0,self.bags):
            learner = self.learner(**self.kwargs)
            n_prime = np.array(np.random.randint(dataY.shape[0], size=int(dataY.shape[0] * .6)))      
            learner.addEvidence(dataX[n_prime, :], dataY[n_prime])
            self.learners.append(learner)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        results = np.zeros(points.shape[0])
        i = 0
        for model in self.learners:
            results += (model.query(points))
            i += 1
        print(i)
        return results / float(i)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
