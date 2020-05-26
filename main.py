import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import PseudoEnv
from mediator import Mediator
from utils import *

env = gym.make("gym_factored:river-swim-v0")

def learn_online(env, agent, mediator):
    state = env.reset()
    cum_reward = 0
    for i in range(MAX_EPISODES):

        #### Start new ####
        # agent_model = agent.learned_model()
        # action = mediator.select_action(state, agent_model)
        ##### End new #####

        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        cum_reward += reward
        agent.process_experience(state, action, new_state, reward, done)
        agent.value_iteration(MAX_ITERATIONS, DELTA)
        state = new_state
        if i % 100 == 0:
            print('Iteration', i, '\t', agent.max_policy(), '\t', cum_reward)
    return agent.Q, cum_reward

rmax = 10000
# c = 0.02
# # c = 0.4
# beta = c * env.rmax
# agent = MBIE_EB(env, beta, gamma)

A = 0.3
B = 0
agent = MBIE(env, maxsize, B, A)

# agent = MBIE(env, 10, 0.5, 0.5)
# agent = MBIE_EB(env, 40000, 0.95)

#### Start new ####
expert_model = PseudoEnv(env) # Initializes expert env (=model).
expert = BoundedParameterExpert(expert_model) # Intializes alg for dealing with model.
mediator = Mediator(expert_model) # Start value iterating in expert model only if iterate = True is passed.
##### End new #####

print(learn_online(env, agent, mediator))
