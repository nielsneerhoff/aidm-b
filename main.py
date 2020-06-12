import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from mediator import Mediator
from utils import *
from metrics import Metrics, write_metrics_to_file

def learn_online(env, agent, metrics, mediator):
    for run in range(NO_RUNS):
        agent.reset()
        state = env.reset()
        metrics.start_runtime(run)
        for i in range(MAX_EPISODES):
            action = mediator.select_action(state)
            new_state, reward, done, info = env.step(action)
            print(state, action, new_state)
            # cum_reward += reward
            T_low_s_a, T_high_s_a = agent.process_experience(
                state, action, new_state, reward, mediator)
            mediator.process_experience(
                state, action, T_low_s_a, T_high_s_a)
            metrics.update_metrics(run, state, action, reward, i)
            state = new_state
            # agent.value_iteration(MAX_ITERATIONS, DELTA)
            if i % 1000 == 0:
                print('Iteration', i, 'run', run, '\t', agent.max_policy(), '\n', agent.Q)
        metrics.calculate_sample_complexity(run)
    return agent.Q #, cum_reward

# Initialize problem env.
env = gym.make("gym_factored:river-swim-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)

# Initialize agents.
mbie_agent = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)

# Initialize expert model & mediator.
expert_model = OffsetModel.from_env(env, 0.1)
mediator = Mediator(expert_model, rho = 0.3)

# Initialize metrics for counting.
mbie_metrics = Metrics(mbie_agent, env, 'mbie')
mbie_eb_metrics = Metrics(mbie_eb_agent, env, 'mbie_eb')

# Run.
print(learn_online(env, mbie_agent, mbie_metrics, mediator))
print(learn_online(env, mbie_eb_agent, mbie_eb_metrics))

write_metrics_to_file([mbie_metrics, mbie_eb_metrics], 'rivers-swim-output')
