
import numpy as np
import pandas as pd

class DTLearner(object):

    def __init__(self, leaf_size, verbose = False):
        self.leaf_size = leaf_size
        self.max_depth = 100
        self.current_depth = 0
        pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'mdunn34' # replace tb34 with your Georgia Tech username

    def get_best_split_corr(self, dataX, dataY):
        return np.argmax([np.abs(np.correlate(dataX[:,i], dataY)) for i in range(dataX.shape[1])])
        

    def buildTree(self, dataX, dataY):

        if dataX.shape[0] == 0:
            return np.array([-1, 0, -1, -1])

        if len(set(dataY)) == 1:
            return [-1, dataY[0], -1, -1]

        if dataX.shape[0] <= self.leaf_size:
            if self.leaf_size != 1:
                return np.array([-1, np.mean(dataY), -1, -1])
            else:
                return np.array([-1, dataY[0], -1, -1])

        i = self.get_best_split_corr(dataX, dataY)

        # using median to make it as close to log base 2 height.
        # trying to keep the tree balanced.
        SplitVal = np.median(dataX[:, i])

        # comprehension, if it were
        left_split = [dataX[:,i]<=SplitVal]

        # grabem columns
        right_split = [dataX[:,i]>SplitVal]

        if np.sum(left_split) == dataX[:,i].shape[0]:
            return np.array([-1, np.mean(dataY), -1, -1])

        lefttree = self.buildTree(dataX[left_split], dataY[left_split])
        righttree = self.buildTree(dataX[right_split], dataY[right_split])

        lefttree = np.array(lefttree)
        righttree = np.array(righttree)

        if len(lefttree.shape) == 1:
            number_of_left = 2
        else:
            number_of_left = lefttree.shape[0] + 1
        
        root = [i, SplitVal, 1, number_of_left]

        self.current_depth += 1
        
        return np.vstack((root, np.vstack((lefttree, righttree))))

    def factorize(self, dataX):
        for i in range(dataX.shape[1]):
            if type(dataX[0,i]) == str:
                dataX[:,i] = np.array(pd.factorize(dataX[:,i])[0]).astype(float)
            dataX[:,i] = dataX[:,i].astype(float)
        return dataX

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
        tree_col = int(self.tree[tree_row][0])
        tree_value = self.tree[tree_row][1]
        point_value = point[tree_col]
        left_tree_index = int(self.tree[tree_row][2])
        right_tree_index = int(self.tree[tree_row][3])
       
        if tree_col == -1:
            return tree_value
        if point_value <= tree_value:
            tree_row += left_tree_index
            return self.traverse(point, tree_row)
        else:
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
        result = np.array(result)
        return result.astype(float) 
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
