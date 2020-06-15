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

        safe_action_merged = self.merged_model.safe_action(
            state, (1 - self.rho) * merged_value
        )

        return safe_action_merged


    def random_action(self, state):
        """
        Returns a random action with higher value than
        (1 - rho) * safe_q_value, if it exists.

        """

        max_safe_q_value = self.safe_q_values.max()
        actions = np.arange(0, self.merged_model.nA)
        within_strictness = actions[
            self.safe_q_values >= (1 - self.rho) * max_safe_q_value]
        return np.random.choice(within_strictness)


    def max_policy(self):
        """
        Return max policy of mediator

        """
        
        return np.argmax(self.merged_model.Q_pes, axis=1)
