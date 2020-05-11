import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

ACTOIN0 = 0
ACTION1 = 1
ACTOIN2 = 2
ACTION3 = 3
ACTOIN4 = 4
ACTION5 = 5

class SixArmsEnv(DiscreteEnv):
    """
    this is a SixArmsEnv chain env using the DiscreteEnv class

    for a more general implementation of a chain env checkout gym/envs/toy_text/nchain.py

    """

    def __init__(self):
        ns = 6
        na = 6 
        t = self.get_transition_function(na, ns)
        r = self.get_reward_function(na,ns)

        p = {s: {a: [] for a in range(na)} for s in range(ns)}
        for s in range(ns):
            for a in range(na):
                for new_state in range(ns):
                    transition_prob = t[s, a, new_state]
                    reward = r[s,a,new_state]
                        p[s][a].append((transition_prob, new_state, reward))
        DiscreteEnv.__init__(self, ns, na, p)

    def get_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))

        #State 0
        t[0, ACTION0, 1] = 1
        t[0, ACTION1, 2] = 0.15
        t[0, ACTOIN2, 3] = 0.1
        t[0, ACTION3, 4] = 0.05
        t[0, ACTOIN4, 5] = 0.03
        t[0, ACTION5, 6] = 0.01
   
        t[0, ACTION0, 0] = 0
        t[0, ACTION1, 0] = 0.85
        t[0, ACTOIN2, 0] = 0.9
        t[0, ACTION3, 0] = 0.95
        t[0, ACTOIN4, 0] = 0.97
        t[0, ACTION5, 0] = 0.99

        #State 1
        t[1, ACTION0, 1] = 1
        t[1, ACTION1, 1] = 1
        t[1, ACTOIN2, 1] = 1
        t[1, ACTION3, 1] = 1
        t[1, ACTOIN4, 0] = 1
        t[1, ACTION5, 1] = 1
   
        #State 2
        t[2, ACTION0, 2] = 1
        t[2, ACTION1, 0] = 1
        t[2, ACTOIN2, 2] = 1
        t[2, ACTION3, 2] = 1
        t[2, ACTOIN4, 2] = 1
        t[2, ACTION5, 2] = 1

        #State 3
        t[3, ACTION0, 0] = 1
        t[3, ACTION1, 0] = 1
        t[3, ACTOIN2, 3] = 1
        t[3, ACTION3, 0] = 1
        t[3, ACTOIN4, 0] = 1
        t[3, ACTION5, 0] = 1

        #State 4
        t[4, ACTION0, 0] = 1
        t[4, ACTION1, 0] = 1
        t[4, ACTOIN2, 0] = 1
        t[4, ACTION3, 4] = 1
        t[4, ACTOIN4, 0] = 1
        t[4, ACTION5, 0] = 1

        #State 5
        t[5, ACTION0, 0] = 1
        t[5, ACTION1, 0] = 1
        t[5, ACTOIN2, 0] = 1
        t[5, ACTION3, 0] = 1
        t[5, ACTOIN4, 5] = 1
        t[5, ACTION5, 0] = 1

        #State 6
        t[6, ACTION0, 0] = 1
        t[6, ACTION1, 0] = 1
        t[6, ACTOIN2, 0] = 1
        t[6, ACTION3, 0] = 1
        t[6, ACTOIN4, 0] = 1
        t[6, ACTION5, 6] = 1

        return t

     def get_reward_function(self, na, ns):
        r = np.zeros((ns, na, ns))

        #State 0
        r[0, ACTION0, 1] = 0
        r[0, ACTION1, 2] = 0
        r[0, ACTOIN2, 3] = 0
        r[0, ACTION3, 4] = 0
        r[0, ACTOIN4, 5] = 0
        r[0, ACTION5, 6] = 0

        #State 1
        r[1, ACTION0, 1] = 50
        r[1, ACTION1, 1] = 50
        r[1, ACTOIN2, 1] = 50
        r[1, ACTION3, 1] = 50
        r[1, ACTOIN4, 0] = 0
        r[1, ACTION5, 1] = 50
   
        #State 2
        r[2, ACTION0, 2] = 0
        r[2, ACTION1, 0] = 133
        r[2, ACTOIN2, 2] = 0
        r[2, ACTION3, 2] = 0
        r[2, ACTOIN4, 2] = 0
        r[2, ACTION5, 2] = 0

        #State 3
        r[3, ACTION0, 0] = 0
        r[3, ACTION1, 0] = 0
        r[3, ACTOIN2, 3] = 300
        r[3, ACTION3, 0] = 0
        r[3, ACTOIN4, 0] = 0
        r[3, ACTION5, 0] = 0

        #State 4
        r[4, ACTION0, 0] = 0
        r[4, ACTION1, 0] = 0
        r[4, ACTOIN2, 0] = 0
        r[4, ACTION3, 4] = 800
        r[4, ACTOIN4, 0] = 0
        r[4, ACTION5, 0] = 0

        #State 5
        r[5, ACTION0, 0] = 0
        r[5, ACTION1, 0] = 0
        r[5, ACTOIN2, 0] = 0
        r[5, ACTION3, 0] = 0
        r[5, ACTOIN4, 5] = 1660
        r[5, ACTION5, 0] = 0

        #State 6
        r[6, ACTION0, 0] = 0
        r[6, ACTION1, 0] = 0
        r[6, ACTOIN2, 0] = 0
        r[6, ACTION3, 0] = 0
        r[6, ACTOIN4, 0] = 0
        r[6, ACTION5, 6] = 6000

        return r

    def render(self, mode='human'):
        pass
