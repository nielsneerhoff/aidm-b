import gym
import numpy as np

from agent import MBIE, MBIE_EB

env = gym.make("gym_factored:sixarms-v0")

def learn_online(
    env, agent, max_episodes = 10, gamma = 0.9, max_iterations = 10000, delta = 1e-04):
    for i in range(max_episodes):
        state = env.reset()
        while True:
            old_state = state
            action = agent.select_action(state) ##### Domain-expert
            new_state, reward, done, info = env.step(action)
            agent.process_experience(state, action, new_state, reward, done)
            agent.value_iteration(max_iterations, delta)
            if done:
                break
            state = new_state
    return agent.Q, agent.max_policy()

agent = MBIE_EB(env, 0.9, 0.3)
print(learn_online(env, agent))

def value_iteration(env, max_iterations = 10000, gamma = 0.9, delta = 1e-04):
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
    return Qnew, max_policy(env, Qnew, gamma)

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