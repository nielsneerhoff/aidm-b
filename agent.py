import numpy as np
from numpy import random

from sys import maxsize

from pseudo_env import HighLowModel

from utils import GAMMA, DELTA_R, DELTA_T, MAX_ITERATIONS, DELTA, MAX_EPISODES


class ModelBasedLearner:
    def __init__(self, nS, nA, m, R):

        # Env basics. R_range is tuple of min reward, max reward of env.
        self.nS, self.nA, self.R, self.m = nS, nA, R, m
        self.reward_range = (np.min(R), np.max(R))

        # Stores current state value estimates.
        self.Q_opt = np.zeros((nS, nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((nS, nA, nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.ones((nS, nA, nS)) / (nS)

    def reset(self):
        '''
        Reset the agent to the begin state for next run

        '''

        # Stores current state value estimates.
        self.Q_opt = np.zeros((self.nS, self.nA))

        # Stores # times s, a , s' was observed.
        self.n = np.zeros((self.nS, self.nA, self.nS))

        # Stores transition probability estimates and reward estimates.
        self.T = np.ones((self.nS, self.nA, self.nS)) / (self.nS)


    def process_experience(self, state, action, next_state):
        """
        Update the transition probabilities based on the state, 
        action, and next state.

        """

        # Only update model if within model size, see section 3 bullet point 7.
        if np.sum(self.n[state][action]) < self.m:
            # Increment the # times s, a, s' was observed.
            self.n[state][action][next_state] += 1

            # Adjust mean probability and reward estimate accordingly.
            self.T[state][action] = (self.n[state][action]) / ( np.sum(self.n[state][action]))

    def select_greedy_action(self, state, possibilities = None):
        """
        Returns a greedy action, based on the current state.
        If possibilities is passed, selects an action from this set.

        Make sure possibilities is non-empty, otherwise it will choose the 
        greedy action by default.

        """

        if possibilities is not None and len(possibilities) > 0:
            # best_action = possibilities[
            #     np.argmax(self.Q_opt[state][possibilities])]
            best_action = np.random.choice(possibilities[self.Q_opt[state][possibilities] == np.max(self.Q_opt[state][possibilities])])
        else:
            best_action = np.argmax(self.Q_opt[state])
        return best_action

    def value_iteration(self):
        """
        Perform optimistic value iteration on the current model.
        Follows the algorithm as discussed in

        "An analysis of model-based interval estimation for MDP" 
        by A. Strehl.

        """

        Qnew = np.zeros((self.nS, self.nA))
        for i in range(MAX_ITERATIONS):
            for state in range(self.nS):
                for action in range(self.nA):
                    Qnew[state][action] = self.q_value(state, action)
            if np.abs(np.sum(self.Q_opt) - np.sum(Qnew)) < DELTA:
                break
            self.Q_opt = np.array(Qnew)
        return self.Q_opt

    def max_policy(self):
        """
        Returns the utility-maximizing policy based on current model.

        """

        return np.argmax(self.Q_opt, axis = 1)

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

    def __init__(self, nS, nA, m, R):
        super().__init__(nS, nA, m, R)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        if np.sum(self.n[state][action]) > 0:

            T_max = self.upper_transition_distribution(state, action)
            R = self.R[state][action]

            # Return Q accordingly.
            return R + GAMMA * np.dot(T_max, np.max(self.Q_opt, axis = 1))

        else:
            # Made up version to account for n = 0 initialization.
            return np.sqrt(np.log(2 / DELTA_R) / 2) * (
                self.reward_range[1] - self.reward_range[0]) / (1 - GAMMA)

    def upper_transition_distribution(self, state, action):
        """
        Returns a probability distribution within 1 - delta_t of the mean
        sample distribution that yields the highest expected value for the
        current (state, action) pair.

        """

        T = np.array(self.T[state][action])
        next_states = np.argsort(np.max(self.Q_opt, axis = 1))
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

    def __init__(self, nS, nA, m, beta, R):
        self.beta = beta
        super(MBIE_EB, self).__init__(nS, nA, m, R)

    def q_value(self, state, action):
        """
        Returns the Q estimate of the current state action.

        """

        return self.R[state][action] + GAMMA * np.dot(self.T[state][action], np.max(self.Q_opt, axis = 1)) + self.exploration_bonus(state, action)

    def exploration_bonus(self, state, action):
        """
        Exploration bonus.

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

class Mediator(MBIE):
    """
    Represents the mediator between the expert and agent: the class that 
    selects the actions based on both models.

    """

    def __init__(self, expert_model, rho, select_action_status = ''):
        """
        Sets the properties of the mediator.

        """

        # Copy basic env vars and init superclass.
        nS, nA, R = expert_model.nS, expert_model.nA, expert_model.R
        super().__init__(nS, nA, np.infty, R)

        self.expert_model = expert_model

        # Init pessimistic Q-values.
        self.Q_pes = np.zeros((self.nS, self.nA))
        self.Q_opt = np.zeros((self.nS, self.nA))
        self.Q_pes, self.Q_opt = self.value_iteration()

        # Agent follows 1 - rho opt. pessimistic expert policy.
        self.rho = rho

    def optimistic_q_value(self, state, action):
        """
        Returns the optmistic Q estimate of the current state action.

        """

        T_max = self.upper_transition_distribution(state, action)
        R = self.R[state][action]
        return R + GAMMA * np.dot(T_max, np.max(self.Q_opt, axis = 1))

    def pessimistic_q_value(self, state, action):
        """
        Returns the pessimistic Q estimate of the current state action.

        """

        T_min = self.lower_transition_distribution(state, action)
        R = self.R[state][action]
        return R + GAMMA * np.dot(T_min, np.max(self.Q_pes, axis = 1))

    def select_action(self, state):
        """
        Returns an action to select based on the current merged model, if it 
        has been updated. Otherwise returns a (random/safe) expert action.

        """

        determined_action = self.determined(state)
        if determined_action is not None:
            return determined_action
        else:
            

    def determined(self, state):
        """
        Returns an action that is optimal in both pessimistic and optimistic Q,
        if it exists.

        """

        actions = np.arange(0, self.nA)
        pessimistic_actions = set(actions[
            self.Q_pes[state] == self.Q_pes[state].max()])
        optimistic_actions = set(actions[
            self.Q_opt[state] == self.Q_opt[state].max()])

        determined = pessimistic_actions.intersection(optimistic_actions)
        random_action = np.random.choice(np.array(determined))
        return random_action

    def value_iteration(self):
        """
        Performs 
        -   optimistic vi on agent model (MBIE).
        -   pessimistic vi on merged model (BPMDP).

        Returns Q-values.

        """

        Q_opt = np.zeros((self.nS, self.nA))
        Q_pes = np.zeros((self.nS, self.nA))

        # Optimistic value iteration.
        for i in range(MAX_ITERATIONS):
            for state in range(self.nS):
                for action in range(self.nA):
                    Q_opt[state][action] = self.optimistic_q_value(
                        state, action)
            if i > 1000 and np.abs(np.sum(self.Q_opt) - np.sum(Q_opt)) < DELTA:
                break
            self.Q_opt = np.array(Q_opt)

        # Pessimistic value iteration.
        for i in range(MAX_ITERATIONS):
            for state in range(self.nS):
                for action in range(self.nA):
                    Q_pes[state][action] = self.pessimistic_q_value(
                        state, action)
            if i > 1000 and np.abs(np.sum(self.Q_pes) - np.sum(Q_pes)) < DELTA:
                break
            self.Q_pes = np.array(Q_pes)

        return self.Q_pes, self.Q_opt

    def upper_transition_distribution(self, state, action):
        """
        Returns a probability distribution 
        -   within 1 - delta_t of the mean sample distribution 
        -   within the expert bounds
        that yields the highest expected value for the
        current (state, action) pair.

        """

        T_mean = np.array(self.T[state][action])
        T_high = self.expert_model.T_high[state][action]
        T_low = self.expert_model.T_low[state][action]
        T = T_mean
        next_states = np.argsort(np.max(self.Q_opt, axis = 1))
        desired_next_state = next_states[-1]

        # Clip all probabilities to fit expert lower bounds.
        T = np.maximum(T_mean, T_low)

        # Keep track of which states have removable probability.
        amount_removable = T - T_low
        # amount_removable[desired_next_state] = 0

        # Increment desired next state as most as possible within bounds.
        epsilon_t = self.epsilon_t(state, action)
        T[desired_next_state] = min(
            T_mean[desired_next_state] + epsilon_t / 2, # Within CI.
            T_high[desired_next_state])                 # Within expert bounds.

        next_index = 0
        while np.sum(T) > 1:
            min_next_state = next_states[next_index]
            to_remove = min(
                amount_removable[min_next_state], np.sum(T) - 1)
            T[min_next_state] -= to_remove
            next_index += 1
        return T

    def lower_transition_distribution(self, state, action):
        """
        Returns a probability distribution 
        -   within 1 - delta_t of the mean sample distribution 
        -   within the expert bounds
        that yields the lowest expected value for the
        current (state, action) pair.

        """

        T_mean = np.array(self.T[state][action])
        T_high = self.expert_model.T_high[state][action]
        T_low = self.expert_model.T_low[state][action]
        T = T_mean
        next_states = np.argsort(np.max(self.Q_pes, axis = 1))[::-1][:self.nS]
        desired_next_state = next_states[-1]

        # Clip all probabilities to fit expert lower bounds.
        T = np.maximum(T_mean, T_low)

        # Keep track of which states have removable probability.
        amount_removable = T - T_low
        # amount_removable[desired_next_state] = 0

        # Increment desired next state as most as possible within bounds.
        epsilon_t = self.epsilon_t(state, action)
        T[desired_next_state] = min(
            T_mean[desired_next_state] + epsilon_t / 2, # Within CI.
            T_high[desired_next_state])                 # Within expert bounds.

        next_index = 0
        while np.sum(T) > 1:
            min_next_state = next_states[next_index]
            to_remove = min(
                amount_removable[min_next_state], np.sum(T) - 1)
            T[min_next_state] -= to_remove
            next_index += 1
        return T
