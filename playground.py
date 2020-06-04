import numpy as np

from pseudo_env import HighLowModel

class PlayGround:

    # def upper_transition_distribution(self, state, action):
    def upper_transition_distribution(self, state, action, next_state):
        """
        Finds upper-bound CI probability distribution maximizing expected Q
        value for state and action.

        """

        current_T = self.T[state][action].copy() # Deep copy.
        # max_next_state = np.argmax(np.max(self.Q, axis = 1))
        max_next_state = next_state

        epsilon_t = self.epsilon_t(state, action)
        current_T[max_next_state] += epsilon_t / 2

        removed = 0 # Counts how much probability is removed.
        while removed < epsilon_t / 2 and np.count_nonzero(current_T) > 1:
            min_next_state = None
            min_value = np.inf
            for s, values in enumerate(self.Q):
                if current_T[s] > 0 and np.max(values) < min_value:
                    min_next_state = s
                    min_value = np.max(values)
            remove = np.minimum(current_T[min_next_state], epsilon_t / 2)
            current_T[min_next_state] -= remove
            removed += remove

        return current_T / np.linalg.norm(current_T, 1)

    def lower_transition_distribution(self, state, action, next_state):
        """
        Finds lower-bound CI probability distribution maximizing expected Q
        value for state and action.

        """

        current_T = self.T[state][action].copy() # Deep copy.
        # max_next_state = np.argmax(np.max(self.Q, axis = 1))
        min_next_state = next_state

        epsilon_t = self.epsilon_t(state, action)
        current_T[min_next_state] -= epsilon_t / 2

        removed = 0 # Counts how much probability is removed.
        while removed < epsilon_t / 2 and np.count_nonzero(current_T) > 1:
            min_next_state = None
            min_value = np.inf
            for s, values in enumerate(self.Q):
                if current_T[s] > 0 and np.max(values) < min_value:
                    min_next_state = s
                    min_value = np.max(values)
            remove = np.minimum(current_T[min_next_state], epsilon_t / 2)
            current_T[min_next_state] += remove
            removed -= remove

        return current_T / np.linalg.norm(current_T, 1)

    def learned_model(self):
        """
        Returns a pseudo env as learned by agent. The pseudo env consists of
        lower and upper bounds for the transition probabilities, calculated
        using the epsilon-confidence measure on the current distribution.

        """

        # Else, model is determined by mean plus and minus epsilon.
        T_high = np.ones((self.nS, self.nA, self.nS))
        T_low = np.zeros((self.nS, self.nA, self.nS))

        # Form estimate lower/upper bound for each state, action.
        for state in range(self.nS):
            for action in range(self.nA):

                for next_state in range(self.nS):
                    T_high[state, action, next_state] = \
                        self.upper_transition_distribution(
                            state, action, next_state)
                    T_low[state, action, next_state] = \
                        self.lower_transition_distribution(
                            state, action, next_state)

                    [0, 1, s_1] = [0.2, 0.3, 0.3]
                    [0, 1, s_2] = [0.4, 0.3, 0.3]

                    T_low[0, 1, s] = [0.2, 0.3, 0.3]
        # Take maximum/minimum over all next states.

        # For now assume rewards have no interval.
        return HighLowModel(T_low, T_high, self.R)