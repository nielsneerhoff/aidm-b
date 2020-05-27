import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import PseudoEnv
from metrics import Metrics

env = gym.make("gym_factored:river-swim-v0")

def learn_online(
    env, agent, metrics, max_episodes = 5000, gamma = 0.95, max_iterations = int(1e6), delta = 1e-02):
    state = env.reset()
    for i in range(max_episodes):
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        agent.process_experience(state, action, new_state, reward, done)
        metrics.update_metrics(state, action, reward, i)
        agent.value_iteration(max_iterations, delta)
        if i % 100 == 0:
            print('Iteration', i)
            # print(agent.Q)
            print(metrics)
        state = new_state
    return agent.Q, '\n', agent.max_policy(), '\n', agent.T


agent = MBIE(env, gamma=0.95, m=1e31, A=0.3, B=0.0)
metrics = Metrics(agent)
# agent = MBIE_EB(env, beta=200, gamma=0.95)
print(learn_online(env, agent, metrics))
