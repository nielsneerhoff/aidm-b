import numpy as np

from utils import MAX_ITERATIONS, DELTA, GAMMA

class Expert:
    """
    Represents an expert.

    """

    def __init__(self, env, strictness):

        self.env = env

        # Stores current state value intervals for both modes.
        self.Q_pes = np.ones((env.nS, env.nA, 2)) # Pessimistic.
        self.Q_opt = np.ones((env.nS, env.nA, 2)) # Optimistic.

        # Parameter guaruantees that agent follows 1 - strictness opt. policy.
        self.strictness = strictness

    def value_iteration(self):
        """
        Perform value iteration based on the expert model.

        """

        for i in range(MAX_ITERATIONS):
            self.Q_opt, done_opt = self.optimistic_value_iterate()
            self.Q_pes, done_pes = self.pessimistic_value_iterate()
            if i > 1000 and done_opt and done_pes:
                break
        return self.Q_pes, self.Q_opt

    def optimistic_value_iterate(self):
        """
        Performs one iteration of optimistic value updates. Returns new value intervals for each state, and whether the update difference was smaller than delta.

        """

        # Find the current values (maximum action at states).
        Q_opt_new = np.array(self.Q_opt)
        Q_opt_state_values = np.max(Q_opt_new, axis = 1)

        # Sort on state lb's in increasing order.
        Q_opt_lbs = Q_opt_state_values[:, 0]
        permutation = np.argsort(Q_opt_lbs)
        Q_opt_new[:, :, 0] = self.value_iterate(permutation, Q_opt_lbs)

        # Sort on state ub's in decreasing order.
        Q_opt_ubs = Q_opt_state_values[:, 1]
        permutation = np.argsort(Q_opt_ubs)[::-1][:self.env.nS]
        Q_opt_new[:, :, 1] = self.value_iterate(permutation, Q_opt_ubs)

        return Q_opt_new, np.abs(np.sum(Q_opt_new) - np.sum(self.Q_opt)) < DELTA

    def pessimistic_value_iterate(self):
        """
        Performs one iteration of pessimistic value updates. Returns new value intervals for each state, and whether the update difference was smaller than delta.

        """

        # Find the current values (maximum action at states).
        Q_pes_new = np.array(self.Q_pes)
        Q_pes_state_values = np.max(Q_pes_new, axis = 1)

         # Sort on state ub's in increasing order.
        Q_pes_ubs = Q_pes_state_values[:, 1] ### ADDED

        # Sort on state lb's in decreasing order.
        Q_pes_lbs = Q_pes_state_values[:, 0]
        permutation = np.argsort(Q_pes_lbs)[::-1][:self.env.nS]
        Q_pes_new[:, :, 1] = self.value_iterate(permutation, Q_pes_ubs)

         # Sort on state ub's in increasing order.
        Q_pes_ubs = Q_pes_state_values[:, 1]
        permutation = np.argsort(Q_pes_ubs)
        Q_pes_new[:, :, 0] = self.value_iterate(permutation, Q_pes_lbs)

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

    def safe_q_value(self, state):
        """
        Returns the 'safe' q-value: the optimal pessimistic q-value lower bound.

        """

        return self.Q_pes[state, :, 0].argsort()[-1]