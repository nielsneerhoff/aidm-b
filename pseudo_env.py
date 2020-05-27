import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

class PseudoEnv(DiscreteEnv):
    """
    Represents an environment as given by an expert or agent.

    """

    def __init__(self, env, offset = 0, Ts = None, rewards = None):
        """
        Initializes a pseudo env. For the transition probabilities, we need either an offset, or two transition probability distributions. If no rewards are passed, the expected rewards are calculated from the env in args.

        """

        if offset is 0 and Ts is None and rewards is None:
            print('No offset, transition distribution or rewards provided. Proceeding copy from env.')

        self.nS, self.nA = env.nS, env.nA

        # Domain expert could give expected reward for each state.
        # If not provided, use expected rewards from env.
        if rewards is not None:
            assert rewards.shape == (env.nS, env.nA)
            self.R = np.array(rewards)
        else:
            self.R = self._expected_rewards(env)

        # Expert could give offset, measure of uncertainty on trans. prob.
        # If not provided, use the trans. prob. upper and lower variant.
        if Ts is None:
            self.offset = offset
            self.T_low, self.T_high = self._offset_transition_function(env)
        else:
            self.T_low = Ts[0]
            self.T_high = Ts[1]

    def _expected_rewards(self, env):
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

    def _offset_transition_function(self, env):
        """
        Creates two offset transition probability distributions using arg env.Creates for each s, a, s' tuple a probability minus and plus the offset.

        """

        T = env.get_transition_function(env.nA, env.nS)
        T_low = T.copy()
        T_high = T.copy()
        for state in range(env.nS):
            for action in range(env.nA):
                for next_state in range(env.nS):
                    probability = T[state, action, next_state]
                    T_low[state, action, next_state] = max(probability - self.offset, 0)
                    T_high[state, action, next_state] = min(probability + self.offset, 1)
        return T_low, T_high