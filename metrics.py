class MetricsTracker:

    def __init__(self, agent):
        self.agent = agent
        self.cum_reward = 0
    
    def increase_cum_reward(self, reward):
        self.cum_reward += reward

    def get_cum_reward(self):
        return self.cum_reward

