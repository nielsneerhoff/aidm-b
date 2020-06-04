import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from mediator import Mediator
from utils import *
from metrics import Metrics

env = gym.make("gym_factored:river-swim-v0")

def learn_online(env, agent, mediator, metrics):
    state = env.reset()
    cum_reward = 0
    agent_model = None
    for i in range(MAX_EPISODES):
        action = agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        # print(state, action, new_state)
        cum_reward += reward
        agent.process_experience(state, action, new_state, reward, done)
        # metrics.update_metrics(state, action, reward, i)
        state = new_state
        if i % 101 == 0:
            agent.value_iteration(MAX_ITERATIONS, DELTA)
            print('Iteration', i, '\t', agent.max_policy(), '\n', agent.Q)
            # new_agent_model = agent._learned_model()
            # print('NEW\n', new_agent_model.T_high)
            # old_agent_model = agent.learned_model()
            # print('OLD\n', old_agent_model.T_high)
            # if agent_model.T_high[0, 1, 3] - agent_model.T_low[0, 1, 3] < 0.3:
            #     action = mediator.select_action(state, agent_model)
                # mediator_action = mediator.select_action(state, agent_model)
                # print(metrics_eb)
    metrics.calculate_sample_complexity()
    return agent.Q, cum_reward

# Initialize agents.
m = 1000 # Model size could be infinite.
beta = 4000
mbie_agent = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)

# Intialize expert & mediator.
T = env.get_transition_function(env.nA, env.nS)
T_low = T.copy()
T_low[0, 1, 3] = 0.4
T_high = T.copy()

R = expected_rewards(env)
expert_model = HighLowModel(T_low, T_high, R)

# expert_model = OffsetModel.from_env(env, 0.2)
expert = BoundedParameterExpert(expert_model)
mediator = Mediator(expert)

# Initialize metrics.
metrics_eb = Metrics(mbie_eb_agent, env, 'mbie')

# expert.value_iteration()
print(learn_online(env, mbie_agent, mediator, metrics_eb))
