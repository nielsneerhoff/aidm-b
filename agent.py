import numpy as np

class ModelBasedLearner:
    def __init__(self, env, gamma):

        self.env = env

        # Stores current state value estimates.
        self.Q = np.ones((env.nS, env.nA)) / (1 - gamma) # Optimistic init.

        # Stores # times s, a , s' was observed.
        self.n = np.ones((env.nS, env.nA, env.nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.zeros((env.nS, env.nA, env.nS))
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

    def select_action(self, state):
        """
        Returns an action, selected based on the current state.
 
        """

        return np.argmax(self.Q[state])

    @abstractmethod
    def update_value(self, state, action):
        pass

class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    def __init__(self, env, gamma, m, delta_t, delta_r):

        self.delta_t = delta_t
        self.delta_r = delta_r
        self.m = m
        super().__init__(env, gamma)

    def epsilon_r(self, state, action):
        """
        Returns the epsilon determining confidence interval for the reward distribution (eq. 2 of paper).

        """

        return np.sqrt(np.log(2 / self.delta_r) / (2 * np.sum(self.n[state][action])))

    def epsilon_t(self, state, action):
        """
        Returns the epsilon determining confidence interval for the transition probability distribution (eq. 5 of paper).

        """

        return np.sqrt(
            (2 * np.log(np.power(2, self.env.nS) - 2) - np.log(self.delta_t))
            / self.m)

    def update_value(self, state, action):
        """
        Updates the Q estimate of the current state action.

        """

        # Pick right-tail upper confidence bound on reward.
        max_R = self.R[state][action] + self.epsilon_r(state, action)

        # Find transition probability distribution that maximized expected Q value.
        current_T = np.array(self.T[state][action]) # Deep copy.
        max_next_state = np.argmax(self.Q)
        epsilon_t = self.epsilon_t(state, action)
        current_T[max_next_state] += epsilon_t / 2
        while np.linalg.norm(current_T, 1) > 1:
            min_next_state = np.argmin(self.Q)
            current_T[min_next_state] -= self.T[state][action][min_next_state]
        current_T = current_T / np.linalg.norm(current_T, 1)

        # Update Q accordingly.
        self.Q[state][action] = max_R + np.dot(
            current_T, np.max(self.Q, axis = 1))

class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """

    def __init__(self, env, gamma, beta):
        self.beta = beta
        super().__init__(env, gamma)

    def update_value(self, state, action):
        """
        Updates the Q estimate of the current state action.

        """

        self.Q[state][action] = self.R[state][action] + np.dot(self.T[state][action], np.max(self.Q, axis = 1)) + + self.exploration_bonus(state, action)

    def exploration_bonus(self, state, action):
        return self.beta / np.sqrt(np.sum(self.n[state][action]))