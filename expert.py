import numpy as np

class BoundedParameterExpert:
    def __init__(self, env, gamma):
    
        self.env = env

        # Stores current state value estimates.
        self.Q_low = np.ones((env.nS, env.nA)) # Holds lower bounds.
        self.Q_high = np.ones((env.nS, env.nA)) # Holds upper bounds.

    def value_iteration(self, max_iterations, delta):
        """
        Perform value iteration based on the expert model.

        """

        Qnew = np.array(self.Q)
        for i in range(max_iterations):
            for state in range(self.env.nS):
                for action in range(self.env.nA):
                    Qnew[state][action] = self.q_value(state, action)
            if i > 1000 and abs(np.sum(self.Q) - np.sum(Qnew)) < delta:
                break
            else:
                self.Q = np.array(Qnew)
        return self.Q

    def max_policy(self):
        return np.argmax(self.Q, axis = 1)