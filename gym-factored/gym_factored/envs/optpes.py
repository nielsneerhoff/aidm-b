import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

ACTION0 = 0
ACTION1 = 1


class OptPes(DiscreteEnv):
    """
    this is a OptPes env using the DiscreteEnv class

    for a more general implementation of a chain env checkout gym/envs/toy_text/nchain.py

    """

    def __init__(self):
        self.rmax = 1000000
        ns = 5
        na = 2 
        terminal_states = np.zeros(ns, dtype=bool)
        self.rmax = 200

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
        t[0, ACTION0, 1] = 0.7
        t[0, ACTION0, 2] = 0.3
        t[0, ACTION1, 3] = 0.7  
        t[0, ACTION1, 4] = 0.3


        #State 1
        t[1, ACTION0, 1] = 0.9
        t[1, ACTION0, 0] = 0.1
       
   
        #State 2
        t[2, ACTION0, 2] = 0.9
        t[2, ACTION0, 0] = 0.1
       

        #State 3
        t[3, ACTION0, 3] = 0.9
        t[3, ACTION0, 0] = 0.1
       

        #State 4
        t[4, ACTION0, 4] = 0.9
        t[4, ACTION0, 0] = 0.1
        

        return t

    def get_reward_function(self, na, ns):
        r = np.zeros((ns, na, ns))


        #State 1
        r[1, ACTION0, 1] = 100
        
   
        #State 2
        r[2, ACTION0, 2] = -1000
        
        #State 3
        r[3, ACTION0, 3] = 200

        #State 4
        r[4, ACTION0, 4] = -3000
     
        return r

    def render(self, mode='human'):
        pass
