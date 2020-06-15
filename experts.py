import numpy as np

ACTION1 = 0
ACTION2 = 1
ACTION3 = 2

class ExpertBounds():

    def __init__(self, nS, nA, T_low, T_high, R):
        self.nS, self.nA = nS, nA
        self.T_low = T_low
        self.T_high = T_high
        self.R = R

class SimpleTaxiExpert(ExpertBounds):
    
    
    def __init__(self):
        self.nS = 4
        self.nA = 3
        self.T_low = self.get_lower_transition_function(self.nA, self.nS)
        self.T_high = self.get_higher_transition_function(self.nA, self.nS)
        self.R = self.get_reward_function(self.nA, self.nS)
        
        super().__init__(self.nS, self.nA, self.T_low, self.T_high, self.R)


    def get_lower_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))

        #State 0
        t[0, ACTION1, 1] = 0.60
        t[0, ACTION1, 0] = 0.40
        t[0, ACTION2, 2] = 0.70
        t[0, ACTION2, 0] = 0.30
        t[0, ACTION3, 3] = 0.50
        t[0, ACTION3, 0] = 0.50


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

    def get_higher_transition_function(self, na, ns):
        t = np.zeros((ns, na, ns))

        #State 0
        t[0, ACTION1, 1] = 0.80
        t[0, ACTION1, 0] = 0.20
        t[0, ACTION2, 2] = 0.90
        t[0, ACTION2, 0] = 0.10
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
     

