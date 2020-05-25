from expert import BoundedParameterExpert

class Mediator(BoundedParameterExpert):
    """
    Represents the agent guided by the parameter intervals of the expert.

    """

    def __init__(self, env, gamma):
        super().__init__(env, gamma)