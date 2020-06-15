import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from mediator import Mediator
from utils import *
from metrics import Metrics, write_metrics_to_file
from experts import SimpleTaxiExpert

def learn_online(env, agent, metrics, mediator=None):
    for run in range(NO_RUNS):
        agent.reset()
        state = env.reset()
        metrics.start_runtime(run)
        for i in range(MAX_EPISODES):
            if mediator is not None:
                action = mediator.select_action(state)
            else:
                action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            # print(state, action, new_state)
            # cum_reward += reward
            T_low_s_a, T_high_s_a = agent.process_experience(
                state, action, new_state, reward, mediator)
            if mediator is not None:
                mediator.process_experience(
                    state, action, T_low_s_a, T_high_s_a)
            else:
                agent.value_iteration(MAX_ITERATIONS, DELTA)
            metrics.update_metrics(run, state, action, reward, i)
            state = new_state
            if i % 100 == 0:
                if mediator is not None:
                    print('Iteration', i, 'run', run, '\t')
                else:
                    print('Iteration', i, 'run', run, 'reward', metrics.cumulative_rewards[run, i], '\t', agent.max_policy(), '\n', agent.Q)
        metrics.calculate_sample_complexity(run)
    return agent.Q #, cum_reward

# Initialize problem env.
env = gym.make("gym_factored:river-swim-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)

# Initialize agents.
mbie_agent = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)
mbie_mediator_agent = MBIE(env.nS, env.nA, m, env.reward_range)

# Initialize expert model & mediator.
# expert_model = OffsetModel.from_env(env, 0.2)
expert_model = HighLowModel.from_env(env, [[4, 1, 5, (0.05, 0.5)], [3, 1, 4, (0.05, 0.5)], [2, 1, 3, (0.05, 0.5)]]) # [3, 1, 4, (0.0, 0.3)], [2, 1, 3, (0.0, 0.3)]
mbie_mediator = Mediator(expert_model, rho = 0.3)

# Initialize metrics for counting.
mbie_metrics = Metrics(mbie_agent, env, 'mbie')
mbie_eb_metrics = Metrics(mbie_eb_agent, env, 'mbie_eb')
mbie_mediator_metrics = Metrics(mbie_mediator_agent, env, 'mbie_mediator')

# Run.
# print(learn_online(env, mbie_agent, mbie_metrics))
# print(learn_online(env, mbie_eb_agent, mbie_eb_metrics))
print(learn_online(env, mbie_mediator_agent, mbie_metrics, mbie_mediator))

write_metrics_to_file([mbie_metrics, mbie_eb_metrics, mbie_mediator_metrics], 'rivers-swim-output-3')
