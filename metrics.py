class MetricsTracker:

    def __init__(self, agent):
        self.agent = agent
        self.cum_reward = 0
    
    #Cumulative Reward
    def increase_cum_reward(self, reward):
        self.cum_reward += reward

    def get_cum_reward(self):
        return self.cum_reward

    #Coverage Error
    


def optimal_vi(env, max_iterations = 2, gamma = 0.95, delta = 1e-02):
    Qold = np.zeros(env.nS)
    Qnew = Qold.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                action_values[action] = 0
                for j in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][j]
                    action_values[action] += prob * (reward + gamma * Qold[next_state])
            best_action = np.argmax(action_values)
            Qnew[state] = action_values[best_action]
        if i > 1000 and abs(np.sum(Qold) - np.sum(Qnew)) < delta:
            break
        else:
            Qold = Qnew.copy()
    return optimal_max_policy(env, Qnew, gamma)

def optimal_max_policy(env, Q, gamma):
    best_actions = np.zeros(env.nS)
    for state in range(env.nS):
        action_values = np.zeros(env.nA)
        for action in range(env.nA):
            state_value = 0
            for j in range(len(env.P[state][action])):
                prob, next_state, reward, done = env.P[state][action][j]
                state_action_value = prob * (reward + gamma * Q[next_state])
                state_value += state_action_value
            action_values[action] = state_value
        best_actions[state] = np.argmax(action_values)
    return best_actions


print(optimal_vi(env))



    


