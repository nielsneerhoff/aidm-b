import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

class PseudoEnv(DiscreteEnv):
    """
    Represents an environment as given by an expert.

    """

    def __init__(self, env, offset, rewards = None):

        self.nS = env.nS
        self.nA = env.nA

        # Domain expert could give expected reward for each state.
        if rewards is not None:
            assert rewards.shape == (env.nS, env.nA)
            self.R = np.array(rewards)
        # Else, we use expected rewards from real model, for simplicity.
        else:
            self.R = self.expected_rewards(env)

        # Domain expert gives a measure of uncertainty on trans. prob, called offset.
        self.offset = offset
        T = env.get_transition_function(env.nA, env.nS)
        self.T_low = T.copy()
        self.T_high = T.copy()
        for state in range(env.nS):
            for action in range(env.nA):
                for next_state in range(env.nS):
                    probability = T[state, action, next_state]
                    self.T_low[state, action, next_state] = max(probability - offset, 0)
                    self.T_high[state, action, next_state] = min(probability + offset, 1)

    def expected_rewards(self, env):
        """
        Parses the expected rewards using the environment.

        """

        rewards = np.zeros((env.nS, env.nA))
        for state in range(env.nS):
            for action in env.P[state]:
                for options in env.P[state][action]:
                    for prob, next_state, reward, _ in [options]:
                        rewards[state, action] += prob * reward

        return rewards