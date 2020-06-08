import numpy as np

from expert import BoundedParameterExpert
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

        # Base case.
        expert = self.expert

        if agent_model is not None:
            merged_model = PseudoEnv.merge(agent_model, self.expert.env)
            expert = BoundedParameterExpert(merged_model)
            expert.value_iteration()

            # Pick action that fall within 1 - strictness of pessimistic opt.
            Q_pes = expert.Q_pes[state, :, 0].copy()
            safe_q_value = np.sort(Q_pes)[-1]

            if np.any(Q_pes > safe_q_value):
                action = Q_pes.argmax()
            else:
                actions = np.arange(0, expert.env.nA, 1)
                within_strictness = actions[
                    Q_pes >= (1 - expert.strictness) * safe_q_value]
                action = np.random.choice(within_strictness)
            return action

        return expert.select_action(state)
