"""
Template for implementing QLearner  (c) 2015 Tucker Balch


"""

import numpy as np
import random as rand

class QLearner(object):

    """
    alpha - how much do you trust new information
    gamma - what is the value of future rewards

    until you reach the goal, all rewards are negative.

    essentially because it's getting a negative reward,
    your robot wants to get to the goal immediately.

    State of objective for the learner is to find a policy that
    maximizes reward. In the fewest amount of steps as possible.

    rar, random action rate - if you look at the book, or one of
    the original q-learning papers, early on it should choose random actions.
    if you don't do that, you won't force the learner to explore.

    radr - random action decay rate.

    if you take a look at test_qlearner.py you'll see the numbers for radr and rar

    dyna is set to zero, for dyna the number of dyna updates that you do.
    you'll need to update that when you implement dyna.

    verbose is just going to be false.

    query is the primary function that implements q-learning
    the way it works is we call a test harness.
    This is the new state you're in, sprime, and this is the reward
    you get for the last step you took.

    The steps you need to folow within this funciton, 1st update the q-table.

    you're in a new state and you're getting a reward for the next action you took.

    what was the state I was in before, what action you took beflow

    experience tuple is s - the state you're in, an experience touple, (s, a, s_prime, r)

    query - update the q table, rollymcdice to see if you take a random action. update rar
    but what if you don't take a random action?
    
    what action should I take?

    remember your q-table answers that problem for you.

    you look at the q value for each of the actions you might take.

    there's one other function, querysetstate - 
    you need to remember the last state you were in
    the last action you took. If this is the very first action, you can't
    remember it.

    just remember that was the state you were in

    that is the whole api that you should implement.

    you are supposed to pick a random action with probablity rar

    what about for the actual trader?

    Whe're going to predifine 3 possible actions.

    A lot of success in applying machine learning to a problem involves mapping the problem it solves to the problem that the learner can solve.

    you need to consider what the algorithm can do and map your problem to that.

    how high resolution do you want to consider oyur factors?

    discretizing your factors into an integer so you can plugin your q-learner.
    """
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = True):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.q_table = np.zeros((num_states, num_actions))
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.s = rand.randint(0, num_states-1)
        self.a = 0
        self.dyna = dyna

        if self.dyna is not 0:
        	self.T_count = np.zeros((num_states, num_actions, num_states)) + 0.00001
        	self.T = self.T_count / self.T_count.sum(axis=2, keepdims=True)
        	self.R = -1 * np.ones((num_states, num_actions))

    
    def author(self):
        return "mdunn34"

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.argmax(self.q_table[self.s, :])
        if (rand.uniform(0,1)) > self.rar:
            action = np.argmax(self.q_table[self.s,])
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new reward
        @returns: The selected action
        """
        old_value = ((1. - self.alpha) * self.q_table[self.s, self.a])
        improved_estimate = self.alpha * (r + self.gamma * \
            self.q_table[s_prime, np.argmax(self.q_table[s_prime, :])])
        update = old_value + improved_estimate
        self.q_table[self.s, self.a] = update

        if rand.random() <= self.rar:
        	action = rand.randint(0, self.num_actions-1)
        else:
        	action = np.argmax(self.q_table[s_prime,:])
        
        self.rar -= self.radr

        if self.dyna is not 0:
        	# T' update and R update
        	self.T_count[self.s, self.a, s_prime] += 1
        	self.T = self.T_count / self.T_count.sum(axis=2, keepdims=True)
        	self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + (self.alpha * r)
        	for i in range(0, self.dyna):        	
	        	S = rand.randint(0, self.num_states-1)
	        	A = rand.randint(0, self.num_actions-1)
	        	S_prime = np.argmax(self.T_count[S, A, :])
	        	r = self.R[S, A]
	        	old_dyna_value = ((1. - self.alpha) * self.q_table[S, A])
		        improved_dyna_estimate = self.alpha * (r + self.gamma * \
		            self.q_table[S_prime, np.argmax(self.q_table[S_prime, :])])
		        dyna_update = old_dyna_value + improved_dyna_estimate    	
		        self.q_table[S, A] = dyna_update

        self.a = action
        self.s = s_prime
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
