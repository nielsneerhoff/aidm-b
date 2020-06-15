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
            new_state, reward, done, info = env.step(action)
            agent.process_experience(
                state, action, new_state, reward)
            metrics.update_metrics(run, state, action, reward, i)
            agent.value_iteration()
            state = new_state
            if i % 100 == 0:
                print('Iteration', i, 'run', run, 'reward', metrics.cumulative_rewards[run, i], '\t', agent.max_policy(), '\n', agent.Q)
        metrics.calculate_sample_complexity(run)
    return agent.Q #, cum_reward

# Initialize problem env.
env = gym.make("gym_factored:river-swim-v0")
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)

# Initialize expert model. See pydoc for .from_env function.
expert_model = HighLowModel.from_env(env, [])

# Initialize agents.
mbie = MBIE(env.nS, env.nA, m, env.reward_range)
mbie_eb = MBIE_EB(env.nS, env.nA, m, beta, env.reward_range)
mediator = Mediator(expert_model, rho = 0.3)

# Initialize metrics for counting.
mbie_metrics = Metrics(mbie, env, 'mbie')
mbie_eb_metrics = Metrics(mbie_eb, env, 'mbie_eb')
mediator_metrics = Metrics(mediator, env, 'mediator')

# Run.
print(learn_online(env, mbie_agent, mbie_metrics))
print(learn_online(env, mbie_eb_agent, mbie_eb_metrics))
print(learn_online(env, mediator, mediator_metrics))

write_metrics_to_file([mbie_metrics, mbie_eb_metrics, mediator_metrics], 'rivers-swim-output')
