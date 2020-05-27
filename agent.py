import numpy as np
from numpy import random

from sys import maxsize

from pseudo_env import PseudoEnv

from utils import GAMMA

class ModelBasedLearner:
    def __init__(self, env):

        self.env = env

        # Stores current state value estimates.
        self.Q = np.ones((env.nS, env.nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((env.nS, env.nA, env.nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.ones((env.nS, env.nA, env.nS)) / (env.nS)
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

    def select_action(self, state):
        """
        Returns a greedy action, based on the current state.

        """

        return np.argmax(self.Q[state])

    def value_iteration(self, max_iterations, delta):
        """
        Perform value iteration based on current model.

        """

        Qnew = np.zeros((self.env.nS, self.env.nA))
        for i in range(max_iterations):
            for state in range(self.env.nS):
                for action in range(self.env.nA):
                    Qnew[state][action] = self.q_value(state, action)
            if np.abs(np.sum(self.Q) - np.sum(Qnew)) < delta:
                break
            self.Q = np.array(Qnew)
        return self.Q

    def max_policy(self):
        return np.argmax(self.Q, axis = 1)

    def reset(self):
        self.env.reset()
        if random.uniform(0, 1) > 0.5:
            return 1
        else:
            return 2

    def learned_model(self):
        """
        Returns a pseudo env as learned by agent.

        """

        # If no experience, there is no model.
        if np.sum(self.n) == 0:
            return None

        T_high = np.ones((self.env.nS, self.env.nA, self.env.nS))
        T_low = np.zeros((self.env.nS, self.env.nA, self.env.nS))

        for state in range(self.env.nS):
            for action in range(self.env.nA):
                epsilon_t = self.epsilon_t(state, action)
                T_high[state, action] = np.minimum(
                    self.T[state, action] + epsilon_t, T_high[state][action])
                T_low[state, action] = np.maximum(
                    self.T[state, action] - epsilon_t, T_low[state][action])

        return PseudoEnv(self.env, Ts = [T_low, T_high], rewards = self.R)

    def epsilon_r(self, state, action):
        """
        Returns the epsilon determining confidence interval for the reward distribution (eq. 2 of paper).

        """
        #Delta's used for experiment here delta_r = A
        if np.sum(self.n[state][action]) > 0:
            return self.delta_r * (self.env.rmax/np.sqrt(np.sum(self.n[state][action])))
        return  self.delta_r * self.env.rmax

        # if np.sum(self.n[state][action]) > 0:
        #     return np.sqrt(np.log(2 / self.delta_r) / (2 * np.sum(self.n[state][action])))
        # return  np.sqrt(np.log(2 / self.delta_r) / 2

    def epsilon_t(self, state, action):
        """
        Returns the epsilon determining confidence interval for the transition probability distribution (eq. 5 of paper).

        """

        # Delta's used for experiment here delta_t = B
        if np.sum(self.n[state][action]) > 0:
            return self.delta_t * (1/np.sqrt(np.sum(self.n[state][action])))
        return  self.delta_t

        # return np.sqrt(
        #     (2 * np.log(np.power(2, self.env.nS) - 2) - np.log(self.delta_t))
        #     / self.m)

    def upper_transition_distribution(self, state, action):
        """
        Finds upper-bound CI probability distribution maximizing expected Q value for state and action.

        """

        current_T = self.T[state][action].copy() # Deep copy.
        max_next_state = np.argmax(np.max(self.Q, axis = 1))
        epsilon_t = self.epsilon_t(state, action)
        current_T[max_next_state] += epsilon_t / 2

        removed = 0 # Counts how much probability is removed.
        while removed < epsilon_t / 2 and np.count_nonzero(current_T) > 1:
            min_next_state = None
            min_value = np.inf
            for s, values in enumerate(self.Q):
                if current_T[s] > 0 and np.max(values) < min_value:
                    min_next_state = s
                    min_value = np.max(values)
            remove = np.minimum(current_T[min_next_state], epsilon_t / 2)
            current_T[min_next_state] -= remove
            removed += remove

        return current_T / np.linalg.norm(current_T, 1)

    def lower_transition_distribution(self, state, action):
        """
        Finds lower-bound CI probability distribution minimizing expected Q value for state and action.

        """

        # To do: similar to upper case, but return lower bound.
        pass

class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    def __init__(self, env, m, delta_t, delta_r):
        self.delta_t = delta_t
        self.delta_r = delta_r
        self.m = m
        super().__init__(env)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        # Pick right-tail upper confidence bound on reward.
        epsilon_r = self.epsilon_r(state, action)
        max_R = self.R[state][action] + epsilon_r

        T_high = self.upper_transition_distribution(state, action)

        # Update Q accordingly.
        return max_R + GAMMA * np.dot(T_high, np.max(self.Q, axis = 1))

class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """

    def __init__(self, env, beta):
        super(MBIE_EB, self).__init__(env)
        self.beta = beta

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        # if np.sum(self.n[state][action]) > 0:
        return self.R[state][action] + GAMMA * np.dot(self.T[state][action], np.max(self.Q, axis = 1)) + self.exploration_bonus(state, action)
        # else:
        #     return maxsize / 2

    def exploration_bonus(self, state, action):
        if np.sum(self.n[state][action]) > 0:
            return self.beta / np.sqrt(np.sum(self.n[state][action]))
        return self.beta
        # return self.beta / np.sqrt(np.sum(self.n[state][action]))