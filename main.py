import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import OffsetModel
from mediator import Mediator
from utils import *
from metrics import Metrics

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
        # metrics_eb.update_metrics(state, action, reward, i)
        state = new_state
        if i % 100 == 0:
            agent_model = agent.learned_model()
            # mediator_action = mediator.select_action(state, agent_model)
            print('Iteration', i, '\t', agent.max_policy(), '\n', agent.Q)
            # print(metrics_eb)
    return agent.Q, cum_reward

# Initialize agents.
m = 100 # Model size could be infinite.
beta = 4000
mbie_agent = MBIE(env.nS, env.nA, m, 10000)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta, 10000)

# Intialize expert & mediator.
expert_model = OffsetModel.from_env(env, 0.3)
expert = BoundedParameterExpert(expert_model)
mediator = Mediator(expert)

# Initialize metrics
metrics_eb = Metrics(mbie_eb_agent, env)

# expert.value_iteration()
print(learn_online(env, mbie_agent, mediator))
