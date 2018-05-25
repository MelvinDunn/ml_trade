"""
"""

import numpy as np
from random import randint

class RTLearner(object):

    def __init__(self, leaf_size, verbose = False):
        self.leaf_size = leaf_size
        pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'mdunn34' # replace tb34 with your Georgia Tech username


    def buildTree(self, dataX, dataY):

        if dataX.shape[0] <= 0:
            return np.array([-1, 0, -1, -1])

        if dataX.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(dataY), -1, -1])

        if len(set(dataY)) == 1:
            return [-1, dataY[0], -1, -1]

        i = randint(0, (dataX.shape[1]-1))

        # these ints are very random
        random_value_1 = randint(0, (dataX.shape[0]-1))

        random_value_2 = randint(0, (dataX.shape[0]-1))

        SplitVal =  (dataX[random_value_1,i] + dataX[random_value_2,i]) / 2.

        # comprehension, if it were
        left_split = [dataX[:,i]<=SplitVal]

        # grabem columns
        right_split = [dataX[:,i]>SplitVal]

        lefttree =  self.buildTree(dataX[left_split], dataY[left_split])
        righttree = self.buildTree(dataX[right_split], dataY[right_split])

        lefttree = np.array(lefttree)
        righttree = np.array(righttree)

        if len(lefttree.shape) == 1:
            number_of_left = 2
        else:
            number_of_left = lefttree.shape[0] + 1
        
        root = [i, SplitVal, 1, number_of_left]

        return np.vstack((root, np.vstack((lefttree, righttree))))


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
       
        dataX = np.nan_to_num(dataX)
        # do stuff with the learner
        self.tree = self.buildTree(dataX,dataY)
        # build and save the model

    def traverse(self, point, tree_row=0):
        # column that the tree is using in the point / node
        tree_col = int(self.tree[tree_row][0])
        # value of the specific tree node
        tree_value = self.tree[tree_row][1]
        # the point value the specific tree row is referencing
        point_value = point[tree_col]

        # indices
        left_tree_index = int(self.tree[tree_row][2])
        right_tree_index = int(self.tree[tree_row][3])
           
        if tree_col == -1:
            return tree_value
        elif point_value <= tree_value:
            tree_row += left_tree_index
            return self.traverse(point, tree_row)
        elif point_value > tree_value:
            tree_row += right_tree_index
            return self.traverse(point, tree_row)            
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result = []
        for point in points:
            result.append(self.traverse(point))
        return np.array(result).astype(float)     
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
