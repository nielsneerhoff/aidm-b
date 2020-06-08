import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import Expert
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from mediator import Mediator
from utils import *
from metrics import Metrics, write_metrics_to_file

env = gym.make("gym_factored:optpes-v0")

def learn_online(env, agent, mediator, metrics):
    for run in range(NO_RUNS):
        agent.reset()
        state = env.reset()
        # cum_reward = 0
        agent_model = None
        metrics.start_runtime(run)
        for i in range(MAX_EPISODES):
            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            # print(state, action, new_state)
            # cum_reward += reward
            agent.process_experience(state, action, new_state, reward, done)
            metrics.update_metrics(run, state, action, reward, i)
            state = new_state
            agent.value_iteration(MAX_ITERATIONS, DELTA)
            print('Iteration', i, '\t', agent.max_policy(), '\n', agent.Q)
            action = mediator.select_action(state, agent_model)
            mediator_action = mediator.select_action(state, agent_model)
        metrics.calculate_sample_complexity(run)
    return agent.Q#, cum_reward

# Initialize agents.
m = 1000 # Model size could be infinite.
beta = (1 / (1 - GAMMA)) * np.sqrt(np.log(2 * env.nS * env.nA * m / DELTA) / 2)
mbie_agent = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)

# Intialize expert & mediator.
T = env.get_transition_function(env.nA, env.nS)
T_low = T.copy()
T_low[0, 1, 3] = 0.4
T_high = T.copy()

R = expected_rewards(env)
expert_model = HighLowModel(T_low, T_high, R)
# expert_model = OffsetModel(T, 0.3, R)

# expert_model = OffsetModel.from_env(env, 0.2)
expert = Expert(expert_model, 0.3)
mediator = Mediator(expert)

# Initialize metrics.
# expert.value_iteration()

mbie_metrics = Metrics(mbie_agent, env, 'mbie')

print(learn_online(env, mbie_agent, mediator, mbie_metrics))


mbie_eb_metrics = Metrics(mbie_eb_agent, env, 'mbie_eb')

print(learn_online(env, mbie_eb_agent, mediator, mbie_eb_metrics))


write_metrics_to_file([mbie_metrics, mbie_eb_metrics], 'test')