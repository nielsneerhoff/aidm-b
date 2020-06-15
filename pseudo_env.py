import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from utils import MAX_ITERATIONS, DELTA, GAMMA

class PseudoEnv(DiscreteEnv):
    """
    Represents an environment as given by an expert or agent.

    """

    def __init__(self, nS, nA, T_low, T_high, R):
        """
        Initializes a pseudo env.

        """

        self.nS, self.nA = nS, nA
        self.T_low, self.T_high = T_low, T_high
        self.R = R

        # Compute q-values for current pseudo-env.
        self.improved = True
        self.Q_pes = self.value_iteration()

    def interval_sizes(self, state, action):
        """
        Returns the sizes of the interval of each next state probability.

        """

        return self.T_high[state, action] - self.T_low[state, action]

    def merge(self, s, a, T_low_s_a_, T_high_s_a_):
        """
        Merges self with new transition probabilities if these are tighter.
        Performs new value iteration if new bounds are tighter.

        """

        self.improved = False
        improved = T_high_s_a_ - T_low_s_a_ - self.interval_sizes(s, a) < -1 * DELTA
        self.T_high[s, a][improved] = T_high_s_a_[improved]
        self.T_low[s, a][improved] = T_low_s_a_[improved]

        # If sufficiently tightened, mark as such.
        if np.any(improved):
            self.improved = True

    def value_iteration(self):
        """
        Perform value iteration based on the expert model.

        """

        if self.improved:

            # Init to zero.
            self.Q_pes = np.zeros((self.nS, self.nA))

            for i in range(MAX_ITERATIONS):
                # Find the current values (maximum action at states).
                Q_pes = np.array(self.Q_pes)
                Q_pes_state_values = np.max(Q_pes, axis = 1)

                # Sort on state lb's in increasing order.
                permutation = np.argsort(Q_pes_state_values)
                self.Q_pes = self.value_iterate(
                    permutation, Q_pes_state_values)

                # If converged, break.
                if i > 1000 and np.abs(np.sum(Q_pes) - np.sum(self.Q_pes)) < DELTA:
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

        F = np.zeros((self.nS, self.nA, self.nS))

        # Find order-maximizing/minimizing MDP for permutation.
        for state in range(self.nS):
            for action in range(self.nA):
                used = np.sum(self.T_low[state, action])
                remaining = 1 - used
                for next_state in permutation:
                    minimum = self.T_low[state, action, next_state]
                    desired = self.T_high[state, action, next_state]
                    if desired <= remaining:
                        F[state, action, next_state] = minimum + desired
                    else:
                        F[state, action, next_state] = minimum + remaining
                    remaining = max(0, remaining - desired)

        # Update Q-values using order-maximizing/minimizing MDP.
        Qnew = np.zeros((self.nS, self.nA))
        for p in permutation:
            Qnew[p] = self.R[p] + GAMMA * np.dot(F[p], q_values.T)
        return Qnew

    def best_action_value(self, state):
        """
        Returns the best action and its value for current state.

        """

        best_action = np.argmax(self.Q_pes[state])
        best_value = self.Q_pes[state, best_action]
        return best_action, best_value

    def safe_action(self, state, lb_value):
        """
        Returns a safe action for current state.

        """

        safe = np.arange(0, self.nA)[
            self.Q_pes[state] >= lb_value]
        return np.random.choice(safe)

    def copy(self):
        """Copy the opbject

        :return: new object with same value
        :rtype: PseudoEnv
        """
        return PseudoEnv(self.nS, self.nA, self.T_low.copy(), self.T_high.copy(), self.R.copy())

class HighLowModel(PseudoEnv):
    """
    Represents a pseudo env as defined by a low and high transition probability distribution.

    """

    def __init__(self, T_low, T_high, R):
        nS, nA = R.shape
        super().__init__(nS, nA, T_low, T_high, R)

    @staticmethod
    def from_env(env, args):
        """
        Returns a pseudo-env as determined by arg env and args.

        each arg in args consist of:
            s, a, s' (lb, ub)

        """

        T = env.get_transition_function(env.nA, env.nS)
        T_low = T.copy()
        T_high = T.copy()
        R = expected_rewards(env)
        for arg in args:
            s, a, s_ = arg[0], arg[1], arg[2]
            low, high = arg[3]
            T_low[s, a, s_] = low
            T_high[s, a, s_] = high
        return HighLowModel(T_low, T_high, R)

class OffsetModel(PseudoEnv):
    """
    Represents a pseudo env as defined by a mean transition distribution plus/minus an offset.

    """

    def __init__(self, T, offset, R):
        self.offset = offset
        nS, nA, T_low, T_high = self._offset_transition_function(T)
        super().__init__(nS, nA, T_low, T_high, R)

    def _offset_transition_function(self, T):
        """
        Creates two offset transition probability distributions using arg T.Creates for each s, a, s' tuple a probability minus and plus the offset.

        """

        nS, nA = T.shape[0], T.shape[1]

        T_low = T.copy()
        T_high = T.copy()
        for state in range(nS):
            for action in range(nA):
                for next_state in range(nS):
                    probability = T[state, action, next_state]
                    T_low[state, action, next_state] = max(probability - self.offset, 0)
                    T_high[state, action, next_state] = min(probability + self.offset, 1)
        return nS, nA, T_low, T_high

    @staticmethod
    def from_env(env, offset):
        """
        Returns a pseudo-env as determined by arg env and offset.

        """

        T = env.get_transition_function(env.nA, env.nS)
        R = expected_rewards(env)
        return OffsetModel(T, offset, R)

def expected_rewards(env):
    """
    Parses the expected rewards over next states using the arg env.
    Uses R(s, a, s') from env to compute E_{s'} [ R(s, a) ].

    """

    rewards = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        for action in env.P[state]:
            for options in env.P[state][action]:
                for prob, next_state, reward, _ in [options]:
                    rewards[state, action] += prob * reward
    return rewards