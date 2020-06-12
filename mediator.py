import numpy as np

from expert import Expert
from pseudo_env import PseudoEnv

class Mediator:
    """
    Represents the mediator between the expert and agent: the class that selects the actions based on both models.

    """

    def __init__(self, expert, rho):
        """
        Sets the properties of this mediator.

        """

        self.model = expert.env # Merged model.
        self.expert = expert
        self.merged = expert # Init to expert.
        self.updated = False

        # Guaruantees agent follows 1 - rho opt. pessimistic expert policy.
        self.rho = rho

    def process_experience(self, state, action, T_low_s_a_, T_high_s_a_):
        """
        Processes the experiences of the agent: updates the merged model
        transition probabilities if the result is tighter.

        """

        self.updated = self.model.merge(
            state, action, T_low_s_a_, T_high_s_a_)

    def select_action(self, state):
        """
        Returns an action to select based on the current merged model, if it 
        has been updated. Otherwise returns a (random) expert action.

        """

        # Find safe q-value and action.
        safe_action = self.safe_actions[state]
        safe_q_value = self.safe_q_values[state]

        if self.updated:
            self.merged = Expert(self.model)
            self.updated = False

        merged_action, merged_value = self.merged.best_action_value(state)
        print(merged_value, safe_q_value, merged_value > merged_value)
        if merged_value > self.merged_expert.Q_pes[state][safe_action]:
            return merged_action

        # If merged model (i.e., agent) found higher q-value, pick greedy.
        # if merged_value > safe_q_value:
        #     print(state, merged_action, merged_value)
        #     return merged_action

        # Else, pick action within 1 - rho of expert pessimistic opt.
        return self.expert.safe_action(state, self.rho)

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
