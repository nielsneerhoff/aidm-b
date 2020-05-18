import gym
import numpy as np
from sklearn.preprocessing import normalize

from random import random, choice

class ModelBasedLearner:
    def __init__(self, env):

        # Stores current state value estimates.
        self.Q = np.zeros((env.nS, env.nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((env.nS, env.nA, env.nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.zeros((env.nS, env.nA, env.nS)) / (env.nS)
        self.R = np.zeros((env.nS, env.nA))

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the transition probabilities and rewards based on the state, action, next state and reward.

        """

        # Increment the # times s, a, s' was observed.
        self.n[state][action][next_state] += 1

        # Adjust mean probability and reward estimate accordingly.
        self.T[state][action] = self.n[state][action] / np.sum(self.n[state][action])
        self.R[state][action] = (self.R[state][action] * (np.sum(self.n[state][action]) - 1) + reward) / np.sum(self.n[state][action])
        print(self.R)

        # TO DO: Should also implement eq. 1 t/m 5 of the paper.

class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
 
        """

        # TO DO: This function should implement eq. 7 of the paper.

class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """

    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
 
        """

        # TO DO: This function should implement eq. 8 of the paper.
