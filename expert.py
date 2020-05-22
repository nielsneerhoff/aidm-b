import numpy as np

class BoundedParameterExpert:
    def __init__(self, env, gamma):
    
        self.env = env

        # Stores current state value estimates.
        self.Q_low = np.ones((env.nS, env.nA)) # Holds lower bounds.
        self.Q_high = np.ones((env.nS, env.nA)) # Holds upper bounds.

    def value_iteration(self, max_iterations, delta):
        """
        Perform value iteration based on the expert model.

        """

        Qnew = np.array(self.Q)
        for i in range(max_iterations):
            # Sort lower.
            permutation = np.argsort(self.Q_high)
            value_iterate(permutation)
            if i > 1000 and abs(np.sum(self.Q) - np.sum(Qnew)) < delta:
                break
            else:
                self.Q = np.array(Qnew)

        # Sort upper.
        return self.Q

    def value_iterate(permutation):

        for state in range(self.env.nS):
            for action in range(self.env.nA):
                Qnew[state][action] = self.q_value(state, action)
    def max_policy(self):
        return np.argmax(self.Q, axis = 1)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        # Pick right-tail upper confidence bound on reward.
        max_R = self.R[state][action] + self.epsilon_r(state, action)

        # Find transition probability distribution that maximized expected Q value.
        current_T = np.array(self.T[state][action]) # Deep copy.
        max_next_state = np.argmax(self.Q)
        epsilon_t = self.epsilon_t(state, action)
        current_T[max_next_state] += epsilon_t / 2
        while np.linalg.norm(current_T, 1) > 1:
            min_next_state = np.argmin(self.Q)
            current_T[min_next_state] -= self.T[state][action][min_next_state]
        current_T = current_T / np.linalg.norm(current_T, 1)

        # Update Q accordingly.
        return max_R + np.dot(
            current_T, np.max(self.Q, axis = 1))