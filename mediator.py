import numpy as np

from pseudo_env import PseudoEnv

class Mediator:
    """
    Represents the mediator between the expert and agent: the class that selects the actions based on both models.

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

    def process_experience(self, state, action, T_low_s_a_, T_high_s_a_):
        """
        Processes the experiences of the agent: updates the merged model
        transition probabilities if the result is tighter.

        """

        self.merged_model.merge(state, action, T_low_s_a_, T_high_s_a_)

    def select_action(self, state):
        """
        Returns an action to select based on the current merged model, if it 
        has been updated. Otherwise returns a (random) expert action.

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

    def max_policy(self):
        """
        Return max policy of mediator

        """
        
        return np.argmax(self.merged_model.Q_pes, axis=1)
