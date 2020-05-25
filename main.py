import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import PseudoEnv

env = gym.make("gym_factored:river-swim-v0")

def learn_online(
    env, agent, max_episodes = 5000, gamma = 0.95, max_iterations = 1000000, delta = 0.01):
    state = env.reset()
    cum_reward = 0
    for i in range(max_episodes):
        action = expert.select_action(state, mode = 'pessimistic')
        new_state, reward, done, info = env.step(action)
        agent.process_experience(state, action, new_state, reward, done)
        cum_reward += reward
        # if i % 100 == 0:
        agent.value_iteration(max_iterations, delta)
        print('Iteration', i, '\t', agent.Q, '\t', cum_reward)
        state = new_state
    return agent.Q, cum_reward

gamma = 0.95

rmax = 10000
# c = 0.02
# # c = 0.4
# beta = c * env.rmax
# agent = MBIE_EB(env, beta, gamma)

A = 0.3
B = 0
agent = MBIE(env, gamma, maxsize, B, A)

agent = MBIE(env, 0.95, 10, 0.5, 0.5)
# agent = MBIE_EB(env, 40000, 0.95)
pseudo_env = PseudoEnv(env, 0.2)
expert = BoundedParameterExpert(pseudo_env, 0.95)
print(learn_online(env, agent, expert))
