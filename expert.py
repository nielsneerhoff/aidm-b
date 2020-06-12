import numpy as np

from utils import MAX_ITERATIONS, DELTA, GAMMA

class Expert:
    """
    Represents an expert.

    """

    def __init__(self, env):

        self.env = env

        # Stores current state value intervals.
        self.Q_pes = np.ones((env.nS, env.nA))

    def value_iteration(self):
        """
        Perform value iteration based on the expert model.

        """

        for i in range(MAX_ITERATIONS):
            self.Q_pes, done = self.pessimistic_value_iterate()
            if i > 1000 and done:
                break
        return self.Q_pes

    def pessimistic_value_iterate(self):
        """
        Performs one iteration of pessimistic value updates. Returns new value intervals for each state, and whether the update difference was smaller than delta.

        """

        # Find the current values (maximum action at states).
        Q_pes_new = np.array(self.Q_pes)
        Q_pes_state_values = np.max(Q_pes_new, axis = 1)

        # Sort on state lb's in increasing order.
        permutation = np.argsort(Q_pes_state_values)
        Q_pes_new = self.value_iterate(
            permutation, Q_pes_state_values)

        return Q_pes_new, np.abs(np.sum(Q_pes_new) - np.sum(self.Q_pes)) < DELTA

    def value_iterate(self, permutation, q_values):
        """
        Perform one value iteration based on permutation. Returns numpy array with for each state action pair the estimated Q value.

        """

        F = np.zeros((self.env.nS, self.env.nA, self.env.nS))

        # Find order-maximizing/minimizing MDP for permutation.
        for state in range(self.env.nS):
            for action in range(self.env.nA):
                used = np.sum(self.env.T_low[state, action])
                remaining = 1 - used
                for next_state in permutation:
                    minimum = self.env.T_low[state, action, next_state]
                    desired = self.env.T_high[state, action, next_state]
                    if desired <= remaining:
                        F[state, action, next_state] = minimum + desired
                    else:
                        F[state, action, next_state] = minimum + remaining
                    remaining = max(0, remaining - desired)

        # Update Q-values using order-maximizing/minimizing MDP.
        Qnew = np.zeros((self.env.nS, self.env.nA))
        for p in permutation:
            Qnew[p] = self.env.R[p] + GAMMA * np.dot(F[p], q_values.T)
        return Qnew

    def best_action_value(self, state):
        """
        Returns the best action and its value for current state.

        """

        best_action = np.argmax(self.Q_pes[state])
        best_value = self.Q_pes[state, best_action]
        return best_action, best_value

    def random_action(self, state, q_value, rho):
        """
        Returns a random action with higher value than (1 - rho) * q_value, if it exists.

        """

        actions = np.arange(0, self.env.nA, 1)
        within_strictness = actions[
            self.Q_pes[state] >= (1 - rho) * q_value]
        return np.random.choice(within_strictness)
