import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

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

    def compare(self, state, action, other_pseudo_env):
        """
        

        """

        return self.T_low[state, action]

    @staticmethod
    def merge(agent_model, expert_model):
        """
        Returns a merged pseudo-env using tighest bounds for transition probabilities.

        We assume the expert model has the reward correct.

        """

        # Else, model is determined by mean plus and minus epsilon.
        T_high = np.array((agent_model.T_high))
        T_low = np.array((agent_model.T_low))
        agent_size = agent_model.T_high - agent_model.T_low
        expert_size = expert_model.T_high - expert_model.T_low

        # Form estimate lower/upper bound for each state, action.
        for s in range(agent_model.nS):
            for a in range(agent_model.nA):
                for s_ in range(agent_model.nS):
                    if agent_size[s, a, s_] > expert_size[s, a, s_]:
                        T_high[s, a, s_] = expert_model.T_high[s, a, s_]
                        T_low[s, a, s_] = expert_model.T_low[s, a, s_]

        # Ignore reward for now.
        return HighLowModel(T_low, T_high, expert_model.R)

class HighLowModel(PseudoEnv):
    """
    Represents a pseudo env as defined by a low and high transition probability distribution.

    """

    def __init__(self,nS, nA, T_low, T_high, R):
        super().__init__(nS, nA, T_low, T_high, R)

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