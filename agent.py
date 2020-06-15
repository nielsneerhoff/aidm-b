import numpy as np
from numpy import random

from sys import maxsize

from pseudo_env import HighLowModel

from utils import GAMMA, DELTA_R, DELTA_T, MAX_ITERATIONS, DELTA, MAX_EPISODES


class ModelBasedLearner:
    def __init__(self, nS, nA, m, R_range):

        # Env basics. R_range is tuple of min reward, max reward of env.
        self.nS, self.nA = nS, nA
        self.R_range = R_range

        self.m = m

        # Stores current state value estimates.
        self.Q = np.zeros((nS, nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((nS, nA, nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.ones((nS, nA, nS)) / (nS)
        self.R = np.zeros((nS, nA))

    def reset(self):
        '''
        Reset the agent to the begin state for next run

        '''
        # Stores current state value estimates.
        self.Q = np.zeros((self.nS, self.nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((self.nS, self.nA, self.nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.ones((self.nS, self.nA, self.nS)) / (self.nS)
        self.R = np.zeros((self.nS, self.nA))

    def process_experience(self, state, action, next_state, reward):
        """
        Update the transition probabilities and rewards based on the state, 
        action, next state and reward.

        Returns a pseudo env as learned by agent. The pseudo env consists of
        lower and upper bounds for the transition probabilities, calculated
        using the epsilon-confidence measure on the current distribution.

        """

        # Only update model if within model size, see section 3 bullet point 7.
        if np.sum(self.n[state][action]) < self.m:
            # Increment the # times s, a, s' was observed.
            self.n[state][action][next_state] += 1

            # Adjust mean probability and reward estimate accordingly.
            self.T[state][action] = (self.n[state][action]) / ( np.sum(self.n[state][action]))
            self.R[state][action] = (self.R[state][action] * (np.sum(self.n[state][action]) - 1) + reward) / np.sum(self.n[state][action])

        # Return high and low confidence probabilities on s, a transition.
        epsilon_t = self.epsilon_t(state, action)
        T_low_s_a = np.maximum(self.T[state][action] - epsilon_t, 0)
        T_high_s_a = np.minimum(self.T[state][action] + epsilon_t, 1)

        return T_low_s_a, T_high_s_a

    def select_greedy_action(self, state):
        """
        Returns a greedy action, based on the current state.

        """

        return np.argmax(self.Q[state])

    def value_iteration(self):
        """
        Perform value iteration on current model.

        """

        Qnew = np.array(self.Q)
        for i in range(MAX_ITERATIONS):
            for state in range(self.nS):
                for action in range(self.nA):
                    Qnew[state][action] = self.q_value(state, action)
            if np.abs(np.sum(self.Q) - np.sum(Qnew)) < DELTA:
                break
            self.Q = np.array(Qnew)
        return self.Q

    def max_policy(self):
        """
        Returns the utility-maximizing policy based on current model.

        """

        return np.argmax(self.Q, axis = 1)

    def epsilon_r(self, state, action):
        """
        Returns the epsilon determining confidence interval for the reward
        distribution (eq. 2 of paper). Adapted to fit Hoeffding bound.

        """

        return np.sqrt(
            np.log(2 / DELTA_R) / (2 * np.sum(self.n[state][action]))) * (self.R_range[1] - self.R_range[0])

    def epsilon_t(self, state, action):
        """
        Returns the epsilon determining confidence interval for the transition
        probability distribution (eq. 5 of paper).

        """

        if np.sum(self.n[state][action]) > 0:
            return np.sqrt((2 * np.log(np.power(2, self.nS) - 2) - np.log(DELTA_T)) / np.sum(self.n[state][action]))
        return 1


class MBIE(ModelBasedLearner):
    """
    MBIE agent.

    """

    def __init__(self, nS, nA, m, R_range):
        super().__init__(nS, nA, m, R_range)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        if np.sum(self.n[state][action]) > 0:

            # Pick right-tail upper confidence bound on reward.
            epsilon_r = self.epsilon_r(state, action)
            max_R = self.R[state][action] + epsilon_r

            T_max = self.upper_transition_distribution(state, action)

            # Return Q accordingly.
            return max_R + GAMMA * np.dot(T_max, np.max(self.Q, axis = 1))
        else:
            # return self.R_range[1] / (1 - GAMMA) # See paper below eq. 6.
            # Made up version to account for n = 0 initialization.
            return np.sqrt(np.log(2 / DELTA_R) / 2) * (
                self.R_range[1] - self.R_range[0]) / (1 - GAMMA)

    def upper_transition_distribution(self, state, action):
        """
        Returns a probability distribution within 1 - delta_t of the mean
        sample distribution that yields the highest expected value for the
        current (state, action) pair.

        """

        T = np.array(self.T[state][action])
        next_states = np.argsort(np.max(self.Q, axis = 1))
        desired_next_state = next_states[-1]
        
        # Add epsilon_t to most desired state, remove from others.
        epsilon_t = self.epsilon_t(state, action)
        left_to_remove = np.minimum(
            np.sum(T) - T[desired_next_state], epsilon_t / 2)
        next_index = 0

        # Weight is removed iteratively, starting from most desired next state
        # as this decreased the expected Q-value the least.
        while left_to_remove > 0:
            min_next_state = next_states[next_index]
            remove = np.minimum(T[min_next_state], left_to_remove)
            T[min_next_state] -= remove
            T[desired_next_state] += remove
            left_to_remove -= remove
            next_index += 1
        return T

    def select_action(self, state):
        """
        Returns a greedy action for state.

        """

        return super().select_greedy_action(state)

class MBIE_EB(ModelBasedLearner):
    """
    MBIE-EB agent.

    """

    def __init__(self, nS, nA, m, beta, R_range):
        self.beta = beta
        super(MBIE_EB, self).__init__(nS, nA, m, R_range)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        return self.R[state][action] + GAMMA * np.dot(self.T[state][action], np.max(self.Q, axis = 1)) + self.exploration_bonus(state, action)

    def exploration_bonus(self, state, action):
        """
        Exploration Bonus

        """
        if np.sum(self.n[state][action]) > 0:
            return self.beta / np.sqrt(np.sum(self.n[state][action]))
        else:
            return self.beta
    
    def select_action(self, state):
        """
        Returns a greedy action for state.

        """

        return super().select_greedy_action(state)


class Mediator(ModelBasedLearner):
    """
    Represents the mediator between the expert and agent: the class that 
    selects the actions based on both models.

    """

    def __init__(self, expert_model, rho):
        """
        Sets the properties of this mediator.

        """

        # Init the two models.
        self.expert_model = expert_model
        self.merged_model = self.expert_model.copy()

        # Agent follows 1 - rho opt. pessimistic expert policy.
        self.rho = rho

        # Init superclass.
        super().__init__(expert_model.nS, expert_model.nA, np.infty, expert_model.reward_range)

    def process_experience(self, state, action, next_state, reward):
        """
        Processes the experiences of the agent: updates the merged model
        transition probabilities if the result is tighter.

        """

        # Process experience.
        T_low_s_a_, T_high_s_a_ = super().process_experience(
            state, action, next_state, reward)

        # Merge resulting model with expert model.
        self.merged_model.merge(state, action, T_low_s_a_, T_high_s_a_)

    def select_action(self, state):
        """
        Returns an action to select based on the current merged model, if it 
        has been updated. Otherwise returns a (random/safe) expert action.

        """

        # Find what expert would do.
        best_action, best_value = self.expert_model.best_action_value(state)

        # Find what we would do based on merged model.
        merged_action, merged_value = self.merged_model.best_action_value(state)

        # If expert and merged action are unequal -> return merged action
        if merged_value > self.merged_model.Q_pes[state][best_action]:
            return merged_action

        return self.merged_model.safe_action(
            state, (1 - self.rho) * merged_value)

    def value_iteration(self):
        """
        Performs pessimistic value iteration on the merged model.

        """

        self.Q = self.merged_model.value_iteration()
