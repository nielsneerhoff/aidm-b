import gym
import numpy as np

from agent import MBIE, MBIE_EB
from expert import BoundedParameterExpert
from pseudo_env import PseudoEnv

env = gym.make("gym_factored:river-swim-v0")

def learn_online(
    env, agent, expert, max_episodes = 10000, gamma = 0.95, max_iterations = 1000, delta = 1e-02):
    state = env.reset()
    # Compute the best policy based on expert model.
    expert.value_iteration(max_iterations, delta)
    for i in range(max_episodes):
        action = expert.select_action(state, mode = 'pessimistic')
        new_state, reward, done, info = env.step(action)
        agent.process_experience(state, action, new_state, reward, done)
        agent.value_iteration(max_iterations, delta)
        print('Iteration', i, '\t', agent.max_policy())
        state = new_state
    return agent.Q, agent.max_policy(), agent.T

agent = MBIE(env, 0.95, 10, 0.5, 0.5)
# agent = MBIE_EB(env, 40000, 0.95)
pseudo_env = PseudoEnv(env, 0.2)
expert = BoundedParameterExpert(pseudo_env, 0.95)
print(learn_online(env, agent, expert))