import numpy as np
from numpy import random

from sys import maxsize

class ModelBasedLearner:
    def __init__(self, env, gamma):

        self.env = env

        # Stores gamma
        self.gamma = gamma

        # Stores current state value estimates.
        # self.Q = np.ones((env.nS, env.nA))
        self.Q = np.full((env.nS, env.nA), 1 / (1 - gamma))

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
        Returns an action, selected based on the current state.

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


class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    # def __init__(self, env, m, delta_t, delta_r):
    #     super().__init__(env)
    #     self.m = m
    #     self.delta_t = delta_t
    #     self.delta_r = delta_r

    def __init__(self, env, gamma, m, A, B):
        super().__init__(env, gamma)
        self.m = m
        self.A = A
        self.B = B

    def epsilon_r(self, state, action):
        """
        Returns the epsilon determining confidence interval for the reward distribution (eq. 2 of paper).

        """

        if np.sum(self.n[state][action]) > 0:
            return self.A * (self.env.r_max/np.sqrt(np.sum(self.n[state][action])))
        return  self.A * self.env.r_max

        # if np.sum(self.n[state][action]) > 0:
        #     return np.sqrt(np.log(2 / self.delta_r) / (2 * np.sum(self.n[state][action])))
        # return  np.sqrt(np.log(2 / self.delta_r) / 2

    def epsilon_t(self, state, action):
        """
        Returns the epsilon determining confidence interval for the transition probability distribution (eq. 5 of paper).

        """
        
        if np.sum(self.n[state][action]) > 0:
            return self.B * (1/np.sqrt(np.sum(self.n[state][action])))
        return  self.B

        # return np.sqrt(
        #     (2 * np.log(np.power(2, self.env.nS) - 2) - np.log(self.delta_t))
        #     / self.m)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        # Pick right-tail upper confidence bound on reward.
        epsilon_r = self.epsilon_r(state, action)
        max_R = self.R[state][action] + epsilon_r
            
        # Find CI probability distribution maximizing expected Q value.
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

        current_T = current_T / np.linalg.norm(current_T, 1)

        # Update Q accordingly.
        return max_R + self.gamma * np.dot(current_T, np.max(self.Q, axis = 1))



class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """

    def __init__(self, env, beta, gamma):
        super().__init__(env, gamma)
        self.beta = beta

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        return self.R[state][action] + self.gamma * np.dot(self.T[state][action], np.max(self.Q, axis = 1)) + self.exploration_bonus(state, action)

    def exploration_bonus(self, state, action):
        """
        Returns exploration bonus, beta / n(s,a)

        """

        if np.sum(self.n[state][action]) > 0:
            return self.beta / np.sqrt(np.sum(self.n[state][action]))
        return self.beta
