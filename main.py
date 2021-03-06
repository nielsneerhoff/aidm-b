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
            # print(action)
            new_state, reward, done, info = env.step(action)
            agent.process_experience(
                state, action, new_state, reward)
            metrics.update_metrics(run, state, action, reward, i)
            agent.value_iteration()
            state = new_state
            if i % 100 == 0:
                print('Iteration', i, 'run', run, 'reward', metrics.cumulative_rewards[run, i], '\t', agent.max_policy(), '\n', agent.Q)
        metrics.calculate_sample_complexity(run)
    return agent.Q

# Initialize problem env.
env = gym.make("gym_factored:simpletaxi-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)

# Initialize expert model. See pydoc for .from_env function.
expert_model = HighLowModel.from_env(env, [[0,0,1,(0.6,0.8)],[0,1,2,(0.7,0.9)], [0,1,0,(0.1,0.3)],[0,2,3,(0.5,0.8)]])
# expert_model = HighLowModel.from_env(env, [[4, 1, 5, (0.1, 1)]])

print(expert_model)

# Initialize agents.
mbie = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)
mediator = Mediator(expert_model, rho = 0.05)
mediator_expert_action = Mediator(expert_model, rho = 0.05, select_action_status = 'expert_best_action')
mediator_merged_action = Mediator(expert_model, rho = 0.05, select_action_status = 'merged_best_action')

# Initialize metrics for counting.
mbie_metrics = Metrics(mbie, env, 'mbie')
mbie_eb_metrics = Metrics(mbie_eb, env, 'mbie_eb')
mediator_metrics = Metrics(mediator, env, 'mediator')
mediator_expert_action_metrics = Metrics(mediator_expert_action, env, 'mediator_expert')
mediator_merged_action_metrics = Metrics(mediator_merged_action, env, 'mediator_merged')

# Run.
print(learn_online(env, mbie, mbie_metrics))
print(learn_online(env, mbie_eb, mbie_eb_metrics))
print(learn_online(env, mediator, mediator_metrics))
print(learn_online(env, mediator_expert_action, mediator_expert_action_metrics))
print(learn_online(env, mediator_merged_action, mediator_merged_action_metrics))

metrics_to_print = [
    mbie_metrics, 
    mbie_eb_metrics, 
    mediator_metrics,
    mediator_expert_action_metrics,
    mediator_merged_action_metrics
    ]

write_metrics_to_file(metrics_to_print, 'simpletaxi-test')
