import numpy as np

class ModelBasedLearner:
    def __init__(self, env):

        self.env = env

        # Stores current state value estimates.
        self.Q = np.zeros((env.nS, env.nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((env.nS, env.nA, env.nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.zeros((env.nS, env.nA, env.nS)) / (env.nS)
        self.R = np.zeros((env.nS, env.nA))

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the transition probabilities and rewards based on the state, action, next state and reward.

        """

        # Increment the # times s, a, s' was observed.
        self.n[state][action][next_state] += 1

        # Adjust mean probability and reward estimate accordingly.
        self.T[state][action] = self.n[state][action] / np.sum(self.n[state][action])
        self.R[state][action] = (self.R[state][action] * (np.sum(self.n[state][action]) - 1) + reward) / np.sum(self.n[state][action])
        print(self.R)

        # TO DO: Should also implement eq. 1 t/m 5 of the paper.

class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    def __init__(self, env, m, delta):

        self.CI_T = set()
        # Note: upper-bound of R is just R(s,a) + epsilon
        self.delta = delta
        self.m = m
        super().__init__(env)



    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
 
        """

        # TO DO: This function should implement eq. 7 of the paper.

    def epsilon(self, state, action):
        """
        Returns the epsilon determining confidence interval for s and a.

        """

        return np.sqrt(
            (2 * np.log(np.power(2, self.env.nS) - 2) - np.log(self.delta))
            / self.m)

class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """
    def __init__(self, env, beta):
        super().__init__(env)
        self.beta = beta
        self.Qold_est = np.zeros(env.nS, env.nA)
        self.Qnew_est = Qold_est.copy()



    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
        
        """

        for action in range(env.nA):
            intervalparam = self.beta / sqrt((self.n[state][action]))
            T_hat_com = 0
            for j in range(len(env.T[state][action])):
                Qold_max = np.max(Qold_est[j])
                T_hat_com += self.T[state][action][j]*Qold_max  

            self.Qnew_est[state][action] = self.R[state][action] + self.gamma*T_hat_com + intervalparam 

            self.Qold_est = self.Qnew_est
    
        return np.argmax(Qnew_est[state])

        # TO DO: This function should implement eq. 8 of the paper.
