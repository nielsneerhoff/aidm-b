import gym
import numpy as np

from agent import MBIE, MBIE_EB

env = gym.make("gym_factored:river-swim-v0")

def learn_online(
    env, agent, max_episodes = 5000, gamma = 0.95, max_iterations = int(1e6), delta = 1e-02):
    state = env.reset()
    cumulative_reward = 0
    for i in range(max_episodes):
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        cumulative_reward += reward
        agent.process_experience(state, action, new_state, reward, done)
        agent.value_iteration(max_iterations, delta)
        if i % 100 == 0:
            print('Iteration', i, '\t', agent.max_policy())
            print('Cumulative reward', cumulative_reward)
            print(agent.Q)
        state = new_state
    return agent.Q, '\n', agent.max_policy(), '\n', agent.T, '\n', cumulative_reward 


agent = MBIE(env, gamma=0.95, m=1e31, A=0.3, B=0.0)
# agent = MBIE_EB(env, beta=200, gamma=0.95)
print(learn_online(env, agent))
