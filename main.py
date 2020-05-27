import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import PseudoEnv
from mediator import Mediator
from utils import *

env = gym.make("gym_factored:optpes-v0")

def learn_online(env, agent, mediator):
    state = env.reset()
    cum_reward = 0
    for i in range(MAX_EPISODES):

        #### Start new ####
        # action = mediator.select_action(state, agent_model)
        ##### End new #####

        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        print(state, new_state)
        cum_reward += reward
        agent.process_experience(state, action, new_state, reward, done)
        agent.value_iteration(MAX_ITERATIONS, DELTA)
        state = new_state
        if i % 100 == 0:
            agent_model = agent.learned_model()
            mediator_action = mediator.select_action(state, agent_model)
            print('Iteration', i, '\t', agent.max_policy(), '\t', cum_reward)
    return agent.Q, cum_reward

rmax = 10000
# c = 0.02
# # c = 0.4
# beta = c * env.rmax
# agent = MBIE_EB(env, beta, gamma)

A = 0.3
B = 0
#agent = MBIE(env, maxsize, B, A)

agent = MBIE(env, 10, 0.5, 0.5)
# agent = MBIE_EB(env, 40000, 0.95)

#### Start new ####
Tlow = env.get_transition_function(env.nA, env.nS)
Thigh = Tlow.copy() 
Tlow[0, 1, 3] = 0.4
T = (Tlow, Thigh)

expert_model = PseudoEnv(env, Ts = T) # Initializes expert env (=model).
expert = BoundedParameterExpert(expert_model) # Intializes alg for dealing with model.
mediator = Mediator(expert, iterate = True) # Start value iterating in expert model only if iterate = True is passed.
##### End new #####
# expert.value_iteration()
print(learn_online(env, agent, mediator))
print(expert.select_action(0, "pessimistic"))
print(expert.select_action(0, 'optimistic'))
