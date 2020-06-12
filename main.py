import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB
from expert import Expert
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from mediator import Mediator
from utils import *
from metrics import Metrics, write_metrics_to_file

env = gym.make("gym_factored:river-swim-v0")

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

# Initialize agents.
m = MAX_EPISODES # Model size could be infinite.
beta = ((env.reward_range[1] - env.reward_range[0]) / (1 - GAMMA)) * np.sqrt(np.log(2 * env.nS * env.nA * m / DELTA) / 2) # lemma 7

R = expected_rewards(env)
T = env.get_transition_function(env.nA, env.nS)
expert_model = OffsetModel(T, 0.2, R)
mbie_agent = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb_agent = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)

# Intialize expert & mediator.
T = env.get_transition_function(env.nA, env.nS)
# T_low = T.copy()
# T_low[0, 1, 3] = 0.4
# T_high = T.copy()

expert = Expert(expert_model)
mediator = Mediator(expert, rho = 0.3)

# Initialize metrics.
# expert.value_iteration()

mbie_metrics = Metrics(mbie_agent, env, 'mbie')

print(learn_online(env, mbie_agent, mbie_metrics, mediator))

mbie_eb_metrics = Metrics(mbie_eb_agent, env, 'mbie_eb')

print(learn_online(env, mbie_eb_agent, mbie_eb_metrics))


write_metrics_to_file([mbie_metrics, mbie_eb_metrics], 'rivers-swim-output')
