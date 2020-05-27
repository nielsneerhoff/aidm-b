import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import OffsetModel
from mediator import Mediator
from utils import *

env = gym.make("gym_factored:river-swim-v0")

def learn_online(env, agent, mediator):
    state = env.reset()
    cum_reward = 0
    for i in range(MAX_EPISODES):
        # action = mediator.select_action(state, agent_model)
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        cum_reward += reward
        agent.process_experience(state, action, new_state, reward, done)
        agent.value_iteration(MAX_ITERATIONS, DELTA)
        state = new_state
        if i % 100 == 0:
            agent_model = agent.learned_model()
            mediator_action = mediator.select_action(state, agent_model)
            print('Iteration', i, '\t', agent.max_policy(), '\t', agent.Q)
    return agent.Q, cum_reward

# Initialize agents.
m = 1
beta = 4000
mbie_agent = MBIE(env.nS, env.nA, m)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta)

# Intialize expert & mediator.
expert_model = OffsetModel.from_env(env, 0.3)
expert = BoundedParameterExpert(expert_model)
mediator = Mediator(expert)

# expert.value_iteration()
print(learn_online(env, mbie_eb_agent, mediator))
