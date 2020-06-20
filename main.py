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

# Initialize problem env.
env = gym.make("gym_factored:bridge-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)
R = expected_rewards(env) # Assume we have perfect knowledge of R.

# Initialize expert model. See pydoc for .from_env function.
expert_model = OffsetModel.from_env(env, 0.2)

# # Initialize agents.
# mbie = MBIE(env.nS, env.nA, m, R)
# mbie_metrics = Metrics(mbie, env, 'mbie')
# mbie_eb = MBIE_EB(env.nS, env.nA, m, beta, R)
# mbie_eb_metrics = Metrics(mbie_eb, env, 'mbie_eb')

# Initialize expert.
rho = 0.3
mediator_max_opt = Mediator(expert_model, rho, safe_action_mode = 'max-opt')
mediator_max_opt_metrics = Metrics(mediator_max_opt, env, 'mediator-max-opt')

# print(learn_online(env, mediator_random, mediator_random_metrics))
print(learn_online(env, mediator_max_opt, mediator_max_opt_metrics))

# print(learn_online(env, mbie, mbie_metrics))
# print(learn_online(env, mbie_eb, mbie_eb_metrics))
# print(learn_online(env, mediator_only_expert, mediator_only_expert_metrics))
# print(learn_online(env, mediator_only_merged, mediator_only_merged_metrics))
# print(learn_online(env, mediator, mediator_metrics))

metrics_to_print = [
    mbie_metrics, 
    mbie_eb_metrics, 
    mediator_only_expert_metrics,
    mediator_only_merged_metrics,
    mediator_metrics
    ]

write_metrics_to_file(metrics_to_print, 'simpletaxi-test')
