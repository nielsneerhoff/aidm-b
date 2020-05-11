import gym
import numpy as np

# env = gym.make('gym_factored:sysadmin-v0')
# env = gym.make('gym_factored:stock-trading-v0')
# env = gym.make('gym_factored:bridge-v0')

env = gym.make("gym_factored:chain-v0")
ob = env.reset()
# decoded_state = list(env.decode(ob))
# assert ob == env.encode(*decoded_state)

# while True:
#     action = env.action_space.sample()
#     ob, reward, done, _ = env.step(action)
#     decoded_state = list(env.decode(ob))
#     print(ob, decoded_state)
#     assert ob == env.encode(*decoded_state)
#     if done:
#         break

def value_iteration(env, max_iterations = 100000, gamma = 0.9, delta = 1e-04):
    Qold = np.zeros(env.nS)
    Qnew = Qold.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                state_value = 0
                for j in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][j]
                    state_action_value = prob * (reward + gamma * Qold[next_state])
                    state_value += state_action_value
                action_values[action] = state_value
            print(action_values)
            best_action = np.argmax(action_values)
            Qnew[state] = action_values[best_action]
            print(f'State {state} old value {Qold[state]} new value {Qnew[state]}')
        if i > 1000 and abs(np.sum(Qold) - np.sum(Qnew)) < delta:
            break
        else:
            Qold = Qnew.copy()
    return Qnew

value_iteration(env)