import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB, Mediator
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from utils import *
from metrics import Metrics, write_metrics_to_file
from experts import SimpleTaxiExpert

def learn_online(env, agent, metrics):
    for run in range(NO_RUNS):
        agent.reset()
        state = env.reset()
        state = 1
        metrics.start_runtime(run)
        for i in range(MAX_EPISODES):
            action = agent.select_action(state)
            new_state, reward, _, _ = env.step(action)
            print(state, action, new_state)
            agent.process_experience(
                state, action, new_state)
            metrics.update_metrics(run, state, action, reward, i)
            agent.value_iteration()
            state = new_state
        metrics.calculate_sample_complexity(run)
    return agent.Q_opt

# # Initialize problem env.
# env = gym.make("gym_factored:river-swim-v0")
# m = MAX_EPISODES # Model size could be infinite.
# beta = BETA(env.reward_range, env.nS, env.nA, m)
# R = expected_rewards(env) # Assume we have perfect knowledge of R.

# # Initialize expert model. See pydoc for .from_env function.
# offsets = [0, 0.1, 0.2, 0.3, 0.5, 0.5]
# rhos = [0.02, 0.04, 0.08, 0.16, 0.32]
# expert_model = OffsetModel.from_env(env, 0.2)

# # Initialize expert.
# rho = 0.3
# mediator_max_opt = Mediator(expert_model, rho, safe_action_mode = 'max-opt')
# mediator_max_opt_metrics = Metrics(mediator_max_opt, env, 'mediator-max-opt')

# mediator_random = Mediator(expert_model, rho, safe_action_mode = 'random')
# mediator_random_metrics = Metrics(mediator_max_opt, env, 'mediator-random')

# print(learn_online(env, mediator_max_opt, mediator_max_opt_metrics))
# print(learn_online(env, mediator_random, mediator_random_metrics))

# metrics_to_print = [
#     mbie_metrics, 
#     mbie_eb_metrics, 
#     mediator_only_expert_metrics,
#     mediator_only_merged_metrics,
#     mediator_metrics
#     ]

# write_metrics_to_file(metrics_to_print, 'simpletaxi-test')
