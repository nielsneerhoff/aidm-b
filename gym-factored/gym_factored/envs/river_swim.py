import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

LEFT = 0
RIGHT = 1

def left(s):
    return max(s - 1, 0)

def right(s, ns):
    return min(s + 1, ns - 1)

class RiverSwimEnv(DiscreteEnv):
    """
    This is a simple River Swim environment using the DiscreteEnv class

    """

    def __init__(self):
        ns = 6
        na = 2  # left and right
        # The first and last states are terminal states
        terminal_states = np.zeros(ns, dtype=bool)
        terminal_states[0] = True
        terminal_states[ns - 1] = True

        t = self.get_transition_function(na, ns)

        isd = np.zeros(ns)
        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            if not terminal_states[s]:
                isd[s] = 1
            for a in range(na):
                for new_state in range(ns):
                    transition_prob = t[s, a, new_state]
                    if transition_prob > 0:
                        if new_state == ns - 1:
                            reward = 10000
                        elif new_state == 0:
                            reward = 5
                        else:
                            reward = 0
                        done = terminal_states[new_state]
                        p[s][a].append((transition_prob, new_state, reward, done))
        isd /= isd.sum()
        DiscreteEnv.__init__(self, ns, na, p, isd)

    def get_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))

        # Transition probabilities from non-terminal states
        for s in range(1, ns - 1):
            # If you swim left, you will end up left for sure.
            t[s, LEFT, left(s)] = 1
            # If you swim right, you end up in the same state, the previous state (water resistance), or the next
            t[s, RIGHT, right(s, ns)] = 0.3
            t[s, RIGHT, s] = 0.6
            t[s, RIGHT, left(s)] = 0.1

        # T from state 0 and n - 1 (terminal states)
        t[0, LEFT, left(0)] = 1
        t[0, RIGHT, 0] = 0.7
        t[0, RIGHT, right(0, ns)] = 0.3

        t[ns - 1, LEFT, left(ns - 1)] = 1
        t[ns - 1, RIGHT, left(ns - 1)] = 0.7
        t[ns - 1, RIGHT, ns - 1] = 0.3
        return t

    def render(self, mode='human'):
        pass

    # def step(self, a):
    #     res = super().step(a)
    #     return state, reward, done
