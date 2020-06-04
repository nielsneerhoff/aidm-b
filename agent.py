import numpy as np
from numpy import random

from sys import maxsize

from pseudo_env import HighLowModel

from utils import GAMMA, DELTA_R, DELTA_T

class ModelBasedLearner:
    def __init__(self, nS, nA, m, R_range):

        # Env basics. R_range is tuple of min reward, max reward of env.
        self.nS, self.nA = nS, nA
        self.R_range = R_range

        self.m = m

        # Stores current state value estimates.
        self.Q = np.zeros((nS, nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((nS, nA, nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.ones((nS, nA, nS)) / (nS)
        self.R = np.zeros((nS, nA))

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the transition probabilities and rewards based on the state, 
        action, next state and reward.

        """

        # Only update model if within model size (see section 3 bullet point 7)
        if np.sum(self.n[state][action]) < self.m:
            # Increment the # times s, a, s' was observed.
            self.n[state][action][next_state] += 1

            # Adjust mean probability and reward estimate accordingly.
            self.T[state][action] = (self.n[state][action]) / ( np.sum(self.n[state][action]))
            self.R[state][action] = (self.R[state][action] * (np.sum(self.n[state][action]) - 1) + reward) / np.sum(self.n[state][action])

    def select_action(self, state):
        """
        Returns a greedy action, based on the current state.

        """

        return np.argmax(self.Q[state])

    def value_iteration(self, max_iterations, delta):
        """
        Perform value iteration on current model.

        """

        Qnew = np.array(self.Q)
        for i in range(max_iterations):
            for state in range(self.nS):
                for action in range(self.nA):
                    Qnew[state][action] = self.q_value(state, action)
            if np.abs(np.sum(self.Q) - np.sum(Qnew)) < delta:
                break
            self.Q = np.array(Qnew)
        return self.Q

    def max_policy(self):
        """
        Returns the utility-maximizing policy based on current model.

        """

        return np.argmax(self.Q, axis = 1)

    def learned_model(self):
        """
        Returns a pseudo env as learned by agent. The pseudo env consists of
        lower and upper bounds for the transition probabilities, calculated
        using the epsilon-confidence measure on the current distribution.

        """

        # If no experience, there is no model.
        if np.sum(self.n) == 0:
            return None

        # Else, model is determined by mean plus and minus epsilon.
        T_high = np.ones((self.nS, self.nA, self.nS))
        T_low = np.zeros((self.nS, self.nA, self.nS))

        # Form estimate lower/upper bound for each state, action.
        for state in range(self.nS):
            for action in range(self.nA):
                epsilon_t = self.epsilon_t(state, action)
                T_high[state, action] = np.minimum(
                    self.T[state, action] + epsilon_t, T_high[state][action])
                T_low[state, action] = np.maximum(
                    self.T[state, action] - epsilon_t, T_low[state][action])

        # For now assume rewards have no interval.
        return HighLowModel(T_low, T_high, self.R)

    def epsilon_r(self, state, action):
        """
        Returns the epsilon determining confidence interval for the reward
        distribution (eq. 2 of paper). Adapted to fit Hoeffding bound.

        """

        return np.sqrt(
            np.log(2 / DELTA_R) / (2 * np.sum(self.n[state][action]))) * (self.R_range[1] - self.R_range[0])

    def epsilon_t(self, state, action):
        """
        Returns the epsilon determining confidence interval for the transition
        probability distribution (eq. 5 of paper).

        """

        # Note, I suppose there is a mistake in the paper (equation 5).
        return np.sqrt(
            (2 * np.log(np.power(2, self.nS) - 2) - np.log(DELTA_T))
            / np.sum(self.n[state][action]))

    def upper_transition_distribution(self, state, action):
        """
        Finds upper-bound CI probability distribution maximizing expected Q
        value for state and action.

        """

        current_T = self.T[state][action].copy() # Deep copy.
        max_next_state = np.argmax(np.max(self.Q, axis = 1))
        epsilon_t = self.epsilon_t(state, action)
        # print('epsilon_t', state, action, epsilon_t)
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

    def _upper_transition_distribution(self, state, action, next_state):

        current_T = self.T[state][action].copy() # Deep copy.
        epsilon_t = self.epsilon_t(state, action)
        current_T[next_state] += epsilon_t / 2

        removed = 0 # Counts how much probability is removed.
        while removed < epsilon_t / 2 and np.count_nonzero(current_T) > 1:
            min_next_state = None
            min_value = np.inf
            for s, values in enumerate(self.Q):
                if s != next_state and current_T[s] > 0 and np.max(values) < min_value:
                    min_next_state = s
                    min_value = np.max(values)
            remove = np.minimum(current_T[min_next_state], epsilon_t / 2)
            current_T[min_next_state] -= remove
            removed += remove

        return current_T / np.linalg.norm(current_T, 1)

    def _lower_transition_distribution(self, state, action, next_state):

        current_T = self.T[state][action].copy() # Deep copy.
        epsilon_t = self.epsilon_t(state, action)

        if epsilon_t / 2 < current_T[next_state]:
            # Want to go least to next state, so decrement probability.
            current_T[next_state] -= epsilon_t / 2

            added = 0 # Counts how much probability is removed.
            while added < epsilon_t / 2 and np.count_nonzero(current_T) > 1:
                max_next_state = None
                max_value = -np.inf
                for s, values in enumerate(self.Q):
                    if s != next_state and current_T[s] > 0 and np.max(values) > max_value:
                        max_next_state = s
                        max_value = np.max(values)
                remove = np.minimum(current_T[max_next_state], epsilon_t / 2)
                current_T[max_next_state] -= added
                added += remove

        return current_T / np.linalg.norm(current_T, 1)

    def _learned_model(self):
        """
        Returns a pseudo env as learned by agent. The pseudo env consists of
        lower and upper bounds for the transition probabilities, calculated
        using the epsilon-confidence measure on the current distribution.

        """

        # If no experience, there is no model.
        if np.sum(self.n) == 0:
            return None

        T_high = np.zeros((self.nS, self.nA, self.nS))
        T_low = np.ones((self.nS, self.nA, self.nS))

        # Form estimate lower/upper bound for each state, action.
        for state in range(self.nS):
            for action in range(self.nA):
                for next_state in range(self.nS):
                    epsilon_t = self.epsilon_t(state, action)
                    print(state, action, epsilon_t < 1)
                    if epsilon_t < 1:
                        upper_bounds = np.maximum(
                            self._upper_transition_distribution(state, action, next_state), T_high[state, action])
                        lower_bounds = np.minimum(
                            self._lower_transition_distribution(state, action, next_state), T_low[state, action])
                    else:
                        upper_bounds = np.ones(self.nS)
                        lower_bounds = np.zeros(self.nS)

                    T_high[state, action] = upper_bounds
                    T_low[state, action] = lower_bounds

        # For now assume rewards have no interval.
        return HighLowModel(T_low, T_high, self.R)

class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    def __init__(self, nS, nA, m, R_range):
        super().__init__(nS, nA, m, R_range)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        if np.sum(self.n[state][action]) > 0:

            # Pick right-tail upper confidence bound on reward.
            epsilon_r = self.epsilon_r(state, action)
            max_R = self.R[state][action] + epsilon_r

            T_max = self.upper_transition_distribution(state, action)

            # Return Q accordingly.
            return max_R + GAMMA * np.dot(T_max, np.max(self.Q, axis = 1))
        else:
            # return self.R_range[1] / (1 - GAMMA) # See paper below eq. 6.
            # Made up version to account for n = 0 initialization.
            return np.sqrt(np.log(2 / DELTA_R) / 2) * (
                self.R_range[1] - self.R_range[0]) / (1 - GAMMA)

class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """

    def __init__(self, nS, nA, m, beta, R_range):
        self.beta = beta
        super(MBIE_EB, self).__init__(nS, nA, m, R_range)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        if np.sum(self.n[state][action]) > 0:
            return self.R[state][action] + GAMMA * np.dot(self.T[state][action], np.max(self.Q, axis = 1)) + self.exploration_bonus(state, action)
        else:
            return self.R_range[1] / (1 - GAMMA) # See paper below eq. 6.

    def exploration_bonus(self, state, action):
        return self.beta / np.sqrt(np.sum(self.n[state][action]))
