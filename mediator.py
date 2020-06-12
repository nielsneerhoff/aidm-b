import numpy as np

from expert import Expert
from pseudo_env import PseudoEnv

class Mediator:
    """
    Represents the mediator between the expert and agent: the class that selects the actions based on both models.

    """

    def __init__(self, expert, rho, iterate = True):
        """
        Sets the properties of this mediator.

        """

        self.expert = expert
        if iterate:
            self.expert.value_iteration()

        # Parameter guaruantees that agent follows 1 - rho optimal pessimistic policy.
        self.rho = rho

    def select_action(self, state, agent_model):
        """
        Returns an action to select based on the current agent and/or expert model (if the agent model is provided).

        """

        # Find safe q-value.
        _, safe_q_value = self.expert.best_action_value(state)
        expert = self.expert

        if agent_model is not None:

            # Combine two models: pick tightest T(s, a, s') for each s, a, s'.
            merged_model = PseudoEnv.merge(agent_model, self.expert.env)
<<<<<<< HEAD
            expert = Expert(merged_model)
            expert.value_iteration()
=======
            merged_expert = Expert(merged_model)
            merged_expert.value_iteration()
>>>>>>> 21b1484dfcfdbfaa2a68749567d53d92e79af167

            action, value = expert.best_action_value(state)

            # If merged model (i.e., agent) found higher q-value, greedy.
            if np.any(value > safe_q_value):
                return action

        # Else, pick action within 1 - rho of pessimistic opt.
        return expert.random_action(state, safe_q_value, self.rho)
