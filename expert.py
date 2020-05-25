import numpy as np

class BoundedParameterExpert:
    """
    Represents the agent guided by the parameter intervals of the expert.

    """

    def __init__(self, env, gamma):

        self.env = env
        self.gamma = gamma

        # Stores current state value intervals for both modes.
        self.Q_pes = np.ones((env.nS, 2)) # Pessimistic.
        self.Q_opt = np.ones((env.nS, 2)) # Optimistic.

    def value_iteration(self, max_iterations, delta):
        """
        Perform value iteration based on the expert model.

        """

        for i in range(max_iterations):
            self.Q_opt, done_opt = self.optimistic_value_iterate(delta)
            self.Q_pes, done_pes = self.pessimistic_value_iterate(delta)
            if i > 1000 and done_opt and done_pes:
                break
        return self.Q_pes, self.Q_opt

    def optimistic_value_iterate(self, delta):
        """
        Performs one iteration of optimistic value updates. Returns new value intervals for each state, and whether the update difference was smaller than delta.

        """

        Q_opt_new = np.array(self.Q_opt)

        # Sort on state lb's in increasing order.
        permutation = np.argsort(Q_opt_new[:, 0])
        Q_opt_new[:, 0] = self.value_iterate(permutation, Q_opt_new[:, 0])

        # Sort on state ub's in decreasing order.
        permutation = np.argsort(Q_opt_new[:, 1])[::-1][:self.env.nS]
        Q_opt_new[:, 1] = self.value_iterate(permutation, Q_opt_new[:, 1])

        return Q_opt_new, np.abs(np.sum(Q_opt_new) - np.sum(self.Q_opt)) < delta

    def pessimistic_value_iterate(self, delta):
        """
        Performs one iteration of pessimistic value updates. Returns new value intervals for each state, and whether the update difference was smaller than delta.

        """

        Q_pes_new = np.array(self.Q_pes)

        # Sort on state lb's in decreasing order.
        permutation = np.argsort(Q_pes_new[:, 0])[::-1][:self.env.nS]
        Q_pes_new[:, 1] = self.value_iterate(permutation, Q_pes_new[:, 0])

        # Sort on state ub's in increasing order.
        permutation = np.argsort(Q_pes_new[:, 1])
        Q_pes_new[:, 0] = self.value_iterate(permutation, Q_pes_new[:, 1])

        return Q_pes_new, np.abs(np.sum(Q_pes_new) - np.sum(self.Q_pes)) < delta

    def value_iterate(self, permutation, q_values):
        """
        Perform one value iteration based on permutation.

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
        Qnew = np.array(q_values)
        for p in permutation:
            Qnew[p] = np.max(self.env.R[p] + self.gamma * np.dot(F[p], q_values.T))
        return Qnew

    def max_policy(self, mode):
        if mode == 'optimistic':
            # Return max policy based on optimistic ordering.
            optimistic_policy = None
        elif mode == 'pessimistic':
            # Return max policy based on optimistic ordering.
            pessimistic_policy = None

        # To do: return some nice statistics, e.g. where do the policies overlap.
        return

    def select_action(self, mode):
        if mode == 'optimistic':
            pass
        elif mode == 'pessimistic':
            # Return max policy based on optimistic ordering.
            pessimistic_policy = None