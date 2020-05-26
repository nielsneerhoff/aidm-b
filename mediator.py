from expert import BoundedParameterExpert

class Mediator:
    """
    Represents the mediator between the expert and agent: the class that selects the actions based on both models.

    """

    def __init__(self, expert, iterate = False):
        """
        Sets the properties of this mediator.

        """

        self.expert = expert
        self.expert.value_iteration() if iterate else None

    def select_action(self, agent_model):
        """
        Returns an action to select based on the current agent and/or expert model (if the agent model is provided).

        """

        # Base case.
        if agent_model is not None:
            pass