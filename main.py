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
env = gym.make("gym_factored:simpletaxi-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)
R = expected_rewards(env) # Assume we have perfect knowledge of R.

# Initialize expert model. See pydoc for .from_env function.
expert_model = HighLowModel.from_env(env, [[0,0,1,(0.6,0.8)],[0,1,2,(0.7,0.9)], [0,1,0,(0.1,0.3)],[0,2,3,(0.5,0.8)]])
# expert_model = OffsetModel.from_env(env, 0.15)
# expert_model = HighLowModel.from_env(env, [
#     [4, 1, 5, (0.05, 1)], [3, 1, 4, (0.05, 1)], [2, 1, 3, (0.10, 1)]])

print(expert_model)

# # Initialize agents.
mbie = MBIE(env.nS, env.nA, m, R)
mbie_eb = MBIE_EB(env.nS, env.nA, m, beta, R)
mediator = Mediator(expert_model, rho = 0.02)
mediator_no_exploration = Mediator(
    expert_model, rho = 0.10, select_action_status = 'mediator_no_exploration')

# Initialize metrics for counting.
mbie_metrics = Metrics(mbie, env, 'mbie')
mbie_eb_metrics = Metrics(mbie_eb, env, 'mbie_eb')
mediator_metrics = Metrics(mediator, env, 'mediator')
mediator_no_exploration_metrics = Metrics(mediator_no_exploration, env, 'mediator_no_exploration')

# Run.
print(learn_online(env, mbie, mbie_metrics))
print(learn_online(env, mbie_eb, mbie_eb_metrics))
print(learn_online(env, mediator, mediator_metrics))
print(learn_online(env, mediator_no_exploration, mediator_no_exploration_metrics))

metrics_to_print = [
    mbie_metrics, 
    mbie_eb_metrics, 
    mediator_metrics,
    mediator_no_exploration_metrics
    ]

write_metrics_to_file(metrics_to_print, 'simpletaxi-test')
