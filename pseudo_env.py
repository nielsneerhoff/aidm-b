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
        self.reward_range = np.min(R), np.max(R)

        # Compute q-values for current pseudo-env.
        self.improved = True
        self.Q_pes, self.Q_opt, self.Q_mean = self.value_iteration()

    def interval_sizes(self, state, action):
        """
        Returns the sizes of the interval of each next state probability.

        """

        return self.T_high[state, action] - self.T_low[state, action]

    def check_validity(self):
        """
        Checks the validity of the interval transition distributions. Lower 
        bounds should sum up below 1, and upper bounds above 1.

        """

        assert np.all(
            np.logical_and(np.sum(self.T_low, axis = 2) <= 1, np.sum(self.T_high, axis = 2) >= 1))

    def merge(self, s, a, T_low_s_a_, T_high_s_a_):
        """
        Merges self with new transition probabilities if these are tighter.
        Performs new value iteration if new bounds are tighter.

        """

        self.improved = False
        improved = T_high_s_a_ - T_low_s_a_ - self.interval_sizes(s, a) < -1 * DELTA
        self.T_high[s, a][improved] = T_high_s_a_[improved]
        self.T_low[s, a][improved] = T_low_s_a_[improved]

        self.check_validity()

        # If sufficiently tightened, mark as such.
        if np.any(improved):
            self.improved = True

    def value_iteration(self):
        """
        Perform value iteration based on the expert model.

        """

        # Only value iterate if the bounds have tightened.
        if not self.improved:
            return

        # Init to zero and pessimistic value iterate.
        self.Q_pes = np.zeros((self.nS, self.nA))
        for i in range(MAX_ITERATIONS):
            self.Q_pes, done = self.pessimistic_value_iterate()
            if i > 1000 and done:
                break

        # Init to zero and pessimistic value iterate.
        self.Q_opt = np.zeros((self.nS, self.nA))
        for i in range(MAX_ITERATIONS):
            self.Q_opt, done = self.optimistic_value_iterate()
            if i > 1000 and done:
                break

        # Init to zero and pessimistic value iterate.
        self.Q_mean = np.zeros((self.nS, self.nA))
        # for i in range(MAX_ITERATIONS):
        #     self.Q_mean, done = self.mean_value_iterate()
        #     if i > 1000 and done:
        #         break

        return self.Q_pes, self.Q_opt, self.Q_mean

    def pessimistic_value_iterate(self):
        """
        Performs one iteration of pessimistic value updates. Returns new value 
        intervals for each state, and whether the update difference was 
        smaller than delta.

        """

        # Find the current values (maximum action at states).
        Q_pes_new = np.array(self.Q_pes)
        Q_pes_state_values = np.max(Q_pes_new, axis = 1)

        # Sort on state lb's in increasing order.
        permutation = np.argsort(Q_pes_state_values)
        Q_pes_new = self.value_iterate(
            permutation, Q_pes_state_values)

        return Q_pes_new, np.abs(np.sum(Q_pes_new) - np.sum(self.Q_pes)) < DELTA

    def optimistic_value_iterate(self):
        """
        Performs one iteration of optimistic value updates. Returns new value 
        intervals for each state, and whether the update difference was 
        smaller than delta.

        """

        # Find the current values (maximum action at states).
        Q_opt_new = np.array(self.Q_opt)
        Q_opt_state_values = np.max(Q_opt_new, axis = 1)

        # Sort on state ub's in decreasing order.
        permutation = np.argsort(Q_opt_state_values)[::-1][:self.nS]
        Q_opt_new = self.value_iterate(
            permutation, Q_opt_state_values)

        return Q_opt_new, np.abs(np.sum(Q_opt_new) - np.sum(self.Q_opt)) < DELTA

    def mean_value_iterate(self):
        """
        Performs one iteration of mean value updates. Returns new value 
        intervals for each state, and whether the update difference was 
        smaller than delta.

        """

        # Find the current values (maximum action at states).
        Q_mean_new = np.array(self.Q_mean)
        Q_mean_state_values = np.max(Q_mean_new, axis = 1)

        # Sort on state ub's in decreasing order.
        mean_T = self.T_high - self.T_low / 2

        q_values = Q_mean_state_values
        # Update Q-values using mean interval MDP.
        for p in range(self.nS):
            Q_mean_new[p] = self.R[p] + GAMMA * np.dot(mean_T[p], q_values.T)

        return Q_mean_new, np.abs(np.sum(Q_mean_new) - np.sum(self.Q_mean)) < DELTA

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

        _safe_actions = self.safe_actions(state, lb_value)
        return np.random.choice(_safe_actions)

    def safe_actions(self, state, lb_value):
        """
        Returns a numpy array of safe actions.

        """

        return np.arange(0, self.nA)[
            self.Q_pes[state] >= lb_value]

    def copy(self):
        """Copy the opbject

        :return: new object with same value
        :rtype: PseudoEnv
        """
        return PseudoEnv(self.nS, self.nA, self.T_low.copy(), self.T_high.copy(), self.R.copy())

    def __str__(self):
        """
        Prints the transition distribution.

        """


        output = 'Transition probabilities\n'
        trans = 's a s\' t\n'
        rewards = '\nRewards\ns a r\n'
        for s in range(self.nS):
            for a in range(self.nA):
                if not self.R[s, a] == 0:
                    rewards += f'{s} {a} {self.R[s, a]}\n'
                for s_ in range(self.nS):
                    if not (self.T_low[s, a, s_] == 0 and self.T_high[s, a, s_] == 0):
                        trans += f'{s} {a} {s_} ({self.T_low[s, a, s_]} {self.T_high[s, a, s_]})\n'
        output += trans
        output += rewards
        return output


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