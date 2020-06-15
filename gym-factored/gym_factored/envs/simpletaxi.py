import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv


ACTION1 = 0
ACTION2 = 1
ACTION3 = 2

class SimpleTaxi(DiscreteEnv):
    """
    this is a SimpleTaxi env using the DiscreteEnv class

    """
    def __init__(self):
        self.reward_range = 0, 100
        ns = 4
        na = 3 
        terminal_states = np.zeros(ns, dtype=bool)

        t = self.get_transition_function(na, ns)
        r = self.get_reward_function(na,ns)
        isd = np.zeros(ns)

        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            isd[s] = 1
            for a in range(na):
                for new_state in range(ns):
                    transition_prob = t[s, a, new_state]
                    reward = r[s,a,new_state]
                    done = terminal_states[new_state]
                    p[s][a].append((transition_prob, new_state, reward, done))
        isd /= isd.sum()
        DiscreteEnv.__init__(self, ns, na, p, isd)
    def get_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))

        #State 0
        t[0, ACTION1, 1] = 0.75
        t[0, ACTION1, 0] = 0.25
        t[0, ACTION2, 2] = 0.70
        t[0, ACTION2, 0] = 0.30
        t[0, ACTION3, 3] = 0.8
        t[0, ACTION3, 0] = 0.20


        #State 1
        t[1, ACTION1, 0] = 1
        t[1, ACTION2, 0] = 1
        t[1, ACTION3, 0] = 1
       
   
        #State 2
        t[2, ACTION1, 0] = 1
        t[2, ACTION2, 0] = 1
        t[2, ACTION3, 0] = 1

        #State 3
        t[3, ACTION1, 0] = 1
        t[3, ACTION2, 0] = 1
        t[3, ACTION3, 0] = 1

        return t

    def get_reward_function(self, na, ns):
        r = np.zeros((ns, na, ns))


        r[0, ACTION1, 1] = 100
        r[0, ACTION1, 0] = 0
        r[0, ACTION2, 2] = 100
        r[0, ACTION2, 0] = 0
        r[0, ACTION3, 3] = 100
        r[0, ACTION3, 0] = 0
     
        return r

    def render(self, mode='human'):
        pass
