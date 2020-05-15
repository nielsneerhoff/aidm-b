import numpy as np
from random import random, choice

class ModelBasedLearner:
    def __init__(self, env):
        self.Q = np.zeros((env.nS, env.nA))
        # Stores transition probability estimates and reward estimates.
        self.T = np.zeros((env.nS, env.nA, env.nS))
        self.R = np.zeros((env.nS, env.nA))

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the transition probabilities and rewards based on the state, action, next state and reward.

        """

        # Should implement eq. 1 t/m 5 of the paper.

class MBIE(ModelBasedLearner):
    """
    MBIE agent

    """

    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
 
        """

        # This function should implement eq. 7 of the paper.

class MBIE_EB(ModelBasedLearner):

    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
 
        """

        # This function should implement eq. 8 of the paper.