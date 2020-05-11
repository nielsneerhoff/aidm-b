import gym
import numpy as np

env = gym.make("gym_factored:river_swim-v0")
ob = env.reset()

def value_iteration(env, max_iterations = 100000, gamma = 0.9, delta = 1e-04):
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
    return max_policy(env, Qnew, gamma)

def max_policy(env, Q, gamma):
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

print(value_iteration(env))