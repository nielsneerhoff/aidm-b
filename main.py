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
            if i % 100 == 0:
                print('Iteration', i, 'run', run, 'reward', metrics.cumulative_rewards[run, i], '\t', agent.max_policy(), '\n', agent.Q_opt)
        metrics.calculate_sample_complexity(run)
    return agent.Q_opt

# Initialize problem env.
env = gym.make("gym_factored:river-swim-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)
R = expected_rewards(env) # Assume we have perfect knowledge of R.

# Initialize expert model. See pydoc for .from_env function.
# expert_model = HighLowModel.from_env(env, [[0, 0, 1, (0.6, 0.8)],[0, 1, 2, (0.7, 0.9)], [0, 1, 0, (0.1, 0.3)],[0, 2, 3, (0.5, 0.8)]])
expert_model = OffsetModel.from_env(env, 0.2)
print(expert_model)

# # Initialize agents.
mbie = MBIE(env.nS, env.nA, m, R)
mbie_metrics = Metrics(mbie, env, 'mbie')
mbie_eb = MBIE_EB(env.nS, env.nA, m, beta, R)
mbie_eb_metrics = Metrics(mbie_eb, env, 'mbie_eb')

select_action_status = 'only-expert-actions'
mediator_only_expert = Mediator(
    expert_model, rho = 0.05, select_action_status=select_action_status)
mediator_only_expert_metrics = Metrics(
    mediator_only_expert, env,  'only-expert-actions')

select_action_status = 'only-merged-actions'
mediator_only_merged = Mediator(
    expert_model, rho = 0.05, select_action_status=select_action_status)
mediator_only_merged_metrics = Metrics(
    mediator_only_merged, env, select_action_status)

select_action_status = 'mediator'
mediator = Mediator(
    expert_model, rho = 0.05, select_action_status = select_action_status)
mediator_metrics = Metrics(mediator_only_merged, env, select_action_status)

# Run.
print(learn_online(env, mediator, mediator_metrics))
print(learn_online(env, mbie, mbie_metrics))
print(learn_online(env, mbie_eb, mbie_eb_metrics))
print(learn_online(env, mediator_only_expert, mediator_only_expert_metrics))
print(learn_online(env, mediator_only_merged, mediator_only_merged_metrics))
print(learn_online(env, mediator, mediator_metrics))

metrics_to_print = [
    mbie_metrics, 
    mbie_eb_metrics, 
    mediator_only_expert_metrics,
    mediator_only_merged_metrics,
    mediator_metrics
    ]

write_metrics_to_file(metrics_to_print, 'simpletaxi-test')
