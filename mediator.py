from expert import BoundedParameterExpert

class Mediator:
    """
    Represents the agent guided by the parameter intervals of the expert.

    """

    def __init__(self, expert_model):
        self.expert_model = expert_model

    def select_action(self, agent_model, expert_model):
        """
        Returns an action to select based on the current agent and expert model.

        """
        pass