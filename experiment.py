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
env = gym.make("gym_factored:simpletaxi-v0")
R = expected_rewards(env) # Assume we have perfect knowledge of R.

####################### MBIE-EB #########################
# m = MAX_EPISODES # Model size could be infinite.
# beta = BETA(env.reward_range, env.nS, env.nA, m)

# mbie = MBIE(env.nS, env.nA, m, R)
# metrics = Metrics(mbie, env, 'mbie')
# learn_online(env, mbie, metrics)
# write_metrics_to_file(
#     [metrics], 'simple-taxi', 'mbie')

# mbie_eb = MBIE_EB(env.nS, env.nA, m, beta, R)
# metrics = Metrics(mbie_eb, env, 'mbie-eb')
# learn_online(env, mbie_eb, metrics)
# write_metrics_to_file(
#     [metrics], 'simple-taxi', 'mbie-eb')
########################################################

####################### Mediator ########################
offsets = [0]
for offset in offsets:
    expert_model = OffsetModel.from_env(env, offset)
    rhos = [0.02, 0.04, 0.08, 0.16, 0.32]
    for rho in rhos:

        print(offset, rho)
        # Max-opt mediator.
        mediator_max_opt = Mediator(
            expert_model, rho, safe_action_mode = 'max-opt')
        metrics = Metrics(
            mediator_max_opt, env, 'mediator-max-opt')
        learn_online(env, mediator_max_opt, metrics)
        write_metrics_to_file(
            [metrics], 'simple-taxi/mediator-max-opt',
            prefix = f'mediator-max-opt-{offset}-{rho}')

        # Random mediator.
        mediator_random = Mediator(
            expert_model, rho, safe_action_mode = 'random')
        metrics = Metrics(
            mediator_max_opt, env, 'mediator-random')
        learn_online(env, mediator_random, metrics)
        write_metrics_to_file(
            [metrics], 'simple-taxi/mediator-random',
            prefix = f'mediator-random-{offset}-{rho}')
#########################################################