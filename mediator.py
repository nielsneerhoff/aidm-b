import numpy as np

from expert import Expert
from pseudo_env import PseudoEnv

class Mediator:
    """
    Represents the mediator between the expert and agent: the class that selects the actions based on both models.

    """

    def __init__(self, expert, iterate = True):
        """
        Sets the properties of this mediator.

        """

        self.expert = expert
        if iterate:
            self.expert.value_iteration()

    def select_action(self, state, agent_model):
        """
        Returns an action to select based on the current agent and/or expert model (if the agent model is provided).

        """

        # Find safe q-value.
        safe_q_value = self.expert.safe_q_value(state)

        if agent_model is not None:

            # Combine the two models: we pick tightest T(s, a, s').
            merged_model = PseudoEnv.merge(agent_model, self.expert.env)
            merged_expert = Expert(merged_model)
            merged_expert.value_iteration()

            Q_pes = merged_expert.Q_pes[state, :, 0]

            # If merged model (i.e., agent) found higher q-value, greedy.
            if np.any(Q_pes > safe_q_value):
                # TO DO: Sort on higher bound (if there is a tie!).
                action = Q_pes.argmax()
                return action
        else:
            Q_pes = self.expert.Q_pes[state, :, 0]

        # Else, pick action within 1 - strictness of pessimistic opt.
        actions = np.arange(0, expert.env.nA, 1)
        within_strictness = actions[
            Q_pes >= (1 - expert.strictness) * safe_q_value]
        action = np.random.choice(within_strictness)

        return action
