import gym
import numpy as np
from sys import maxsize

from agent import MBIE, MBIE_EB, Mediator
from pseudo_env import OffsetModel, HighLowModel, expected_rewards
from utils import *
from metrics import Metrics, write_metrics_to_file
from experts import SimpleTaxiExpert
from main import learn_online

# Initialize problem env.
env = gym.make("gym_factored:river-swim-v0")
R = expected_rewards(env) # Assume we have perfect knowledge of R.

####################### MBIE-EB #########################
m = MAX_EPISODES # Model size could be infinite.
beta = BETA(env.reward_range, env.nS, env.nA, m)



rho = 0.16
offset = 0.1
expert_model = OffsetModel.from_env(env, offset)

# Max-opt mediator.
mediator_max_opt = Mediator(
    expert_model, rho, safe_action_mode = 'max-opt')
metrics = Metrics(
    mediator_max_opt, env, 'mediator-max-opt')
learn_online(env, mediator_max_opt, metrics)
write_metrics_to_file(
    [metrics], 'river-swim/mediator-max-opt',
    prefix = f'mediator-max-opt-{offset}-{rho}')