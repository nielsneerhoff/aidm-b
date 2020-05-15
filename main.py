import gym
import numpy as np

env = gym.make("gym_factored:river-swim-v0")

def learn_online(env, agent, max_iterations = 100000, gamma = 0.9):
    for episode in range(num_episodes):
        state = env.reset()

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()
            printing=False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report()
                print("state:", state)

            old_state = state
            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            if printing:
                print("act:", action)
                print("reward=%s" % reward)

            agent.process_experience(state, action, new_state, reward, done)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                agent.set_next_to_terminal_state(old_state)
                env.render()
                agent.report()
                number_of_steps[episode] = t
                break
            state = new_state
    # decoded_state = list(env.decode(ob))
    # assert ob == env.encode(*decoded_state)
    for i in range(max_iterations):
        env.reset()
        ob = env.reset()
        while True:
            action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            # decoded_state = list(env.decode(ob))
            # print(ob, decoded_state)
            # assert ob == env.encode(*decoded_state)
            if done:
                break
    return Qnew, max_policy(env, Qnew, gamma)

print(learn_online(env))

def value_iteration(env, max_iterations = 1, gamma = 0.9, delta = 1e-04):
    Qold = np.zeros(env.nS)
    Qnew = Qold.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                action_values[action] = 0
                for j in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][j]
                    action_values[action] += prob * (reward + gamma * Qold[next_state])
            best_action = np.argmax(action_values)
            Qnew[state] = action_values[best_action]
        if i > 1000 and abs(np.sum(Qold) - np.sum(Qnew)) < delta:
            break
        else:
            Qold = Qnew.copy()
    return Qnew, max_policy(env, Qnew, gamma)

def max_policy(env, Q, gamma):
    best_actions = np.zeros(env.nS)
    for state in range(env.nS):
        action_values = np.zeros(env.nA)
        for action in range(env.nA):
            state_value = 0
            for j in range(len(env.P[state][action])):
                prob, next_state, reward, done = env.P[state][action][j]
                state_action_value = prob * (reward + gamma * Q[next_state])
                state_value += state_action_value
            action_values[action] = state_value
        best_actions[state] = np.argmax(action_values)
    return best_actions