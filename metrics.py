import numpy as np
import time
from utils import *
import pathlib
import os
import sys

class Metrics:
    '''
    Tasked for registering performance of the agents.

    '''

    def __init__(self, agent, env, name):

        # Check name
        if not isinstance(name, str):
            raise TypeError("parameter 'name' is not of type: string")

        # Environment
        self.env = env

        # Agent
        self.agent = agent

        # Name for writing to file
        self.name = name

        # Start Timer
        self.start_time = np.zeros((NO_RUNS))

        # Runtime
        self.runtime = np.zeros((NO_RUNS, MAX_EPISODES))

        # Cumulative reward received
        self.cumulative_rewards = np.zeros((NO_RUNS, MAX_EPISODES))

        # Transition Function Environment
        self.env_T = self.__get_transition_function()

        # Reward distributions
        self.env_mean_reward = np.zeros((self.env.nS, self.env.nA))
        self.env_std_reward = np.zeros((self.env.nS, self.env.nA))
        self.__init_reward_distributions()

        self.rewards_history = np.frompyfunc(list, 0, 1)(np.empty((NO_RUNS, self.env.nS, self.env.nA), dtype=object))
        self.std_reward = np.zeros((NO_RUNS, self.env.nS, self.env.nA))


        # KL Convergence
        self.KL_divergence_T = np.zeros((NO_RUNS, self.env.nS, self.env.nA))
        self.KL_divergence_R = np.zeros((NO_RUNS, self.env.nS, self.env.nA))
        self.KL_divergence_T_sum = np.zeros((NO_RUNS, MAX_EPISODES))
        self.KL_divergence_R_sum = np.zeros((NO_RUNS, MAX_EPISODES))
        self.__init_KL_divergence()

        # Q-value of optimal policy
        self.env_Q = self.__value_iteration_env()
        
        # Max policy env
        self.env_max_policy = self.__get_env_max_policy()

        # Max policy agent
        self.agent_max_policy = np.zeros((NO_RUNS, self.env.nS))

        # Coverage error transition probabilities and expected reward
        self.coverage_error_squared_T = np.zeros((NO_RUNS, MAX_EPISODES))
        self.coverage_error_squared_R = np.zeros((NO_RUNS, MAX_EPISODES))

        # Sample Complexity Metric
        self.sample_complexity = np.zeros((NO_RUNS))

        # Reward and State timeline for instantaneous loss calculation
        self.reward_timeline = np.zeros((NO_RUNS, MAX_EPISODES))
        self.state_timeline = np.zeros((NO_RUNS, MAX_EPISODES), dtype=int)
        self.future_rewards = np.zeros((NO_RUNS, MAX_EPISODES))

        # Instantaneous Loss 
        self.instantaneous_loss = np.zeros((NO_RUNS, MAX_EPISODES))

    def __str__(self):
        '''
        String representation of the object
        Used for the print() function
        '''
        return f'''
Timer: {self.get_runtime()}
Cumulative reward: {self.cumulative_rewards}
Max policy: {self.agent_max_policy}
KL divergence transition distribution:\n{self.KL_divergence_T}
KL divergence reward distribution:\n{self.KL_divergence_R}
Coverage error transition squared: {self.coverage_error_squared_T}
Coverage error reward squared: {self.coverage_error_squared_R}
Sample complexity: {self.sample_complexity}
Hit zero sample complexity after {self.zero_sample_complexity_steps} steps'''


    ### ''' Public Updater Methods ''' ###


    def start_runtime(self, run):
        '''
        Start runtime metric

        '''
        self.start_time[run] = time.time()


    def get_runtime(self, run):
        '''
        Return runtime up until now.

        '''
        return round(time.time() - self.start_time[run], 5)

    
    def update_metrics(self, run, state, action, reward, step):
        '''
        Update from experience.

        '''
        # updates cumulative reward
        self.__update_cumulative_reward(run, step, reward)
        
        # updates history of rewards and standard deviation of rewards
        self.__update_std_reward(run, state, action, reward)
        
        # updates reward and transition error squared
        self.__update_coverage_error_squared(run, step)

        # updates KL divergence metric for transitions and rewards
        self.__update_KL_divergence(run, step, state, action)

        # updates max policy of agent
        self.__update_max_policy_agent(run)

        # update the time line of the rewards for the instantaneous loss
        self.__update_reward_timeline(run, reward, step, state)

        # update passed runtime
        self.__update_runtime(run, step)


    def calculate_sample_complexity(self, run, epsilon=1e03):
        '''
        Update sample complexity
        Computed as the count of differences between best policy and current policy -> 0 is best
        param epsilon : if difference between policies > epsilon then counter plus one

        '''

        # Init future reward
        self.future_rewards[run, MAX_EPISODES - 1] = self.reward_timeline[run, MAX_EPISODES - 1]

        # Reverse calculate future rewards
        for i in reversed(range(MAX_EPISODES - 1)):
            self.future_rewards[run, i] = self.reward_timeline[run, i] + GAMMA * self.future_rewards[run, i + 1]

        Q_old = np.zeros((self.env.nS, self.env.nA))
        for i in reversed(range(MAX_EPISODES)):
            
            # Calculate V_opt "Q_opt" in backwards fashiojn
            Q_new = np.array([[self.env_mean_reward[state, action] + \
                GAMMA * np.dot(self.env_T[state, action], np.max(Q_old, axis = 1)) \
                    for action in range(self.env.nA)] \
                        for state in range(self.env.nS)])

            # Calculate the instant loss from chosing the action at timestep i
            self.instantaneous_loss[run, i] = np.max(Q_new[self.state_timeline[run, i]]) - self.future_rewards[run, i]

            # if instantaneous loss is greater than epsilon, add one to sample complexity
            if self.instantaneous_loss[run, i] > epsilon:
                self.sample_complexity[run] += 1

            Q_old = Q_new

        
    ### ''' Private Initializer Methods ''' ###
            

    def __get_transition_function(self):
        '''
        Recreate transition function.

        '''
        T = np.zeros((self.env.nS, self.env.nA, self.env.nS))
        for state, actions in self.env.P.items():
            for action, probs in actions.items():
                for (prob, new_state, reward, _) in probs:
                    T[state, action, new_state] = prob
        return T


    def __init_reward_distributions(self):
        '''
        Get reward distributions

        '''
        for state, actions in self.env.P.items():
            for action, probs in actions.items():
                self.env_mean_reward[state, action] = sum(reward * prob for (prob, _, reward, _) in probs)
                self.env_std_reward[state, action] = np.sqrt(sum(
                    abs(reward - self.env_mean_reward[state, action])**2 * prob 
                        for (prob, _, reward, _) in probs))

        
    def __init_KL_divergence(self):
        '''
        Initialize KL divergence arrays with values.
        
        '''
        for run in range(NO_RUNS):
            for state in range(self.env.nS):
                for action in range(self.env.nA):
                    self.__update_KL_divergence(run, 0, state, action)


    def __get_env_max_policy(self):
        '''
        Get optimal policy of the environment

        '''
        return np.argmax(self.env_Q, axis = 1)



    def __value_iteration_env(self, delta = 1e-3):
        '''
        Value iterate over the environment until converged

        '''
        Q_old = np.ones((self.env.nS, self.env.nA))
        while True:
            Q_new = np.array([[self.env_mean_reward[state, action] + \
                GAMMA * np.dot(self.env_T[state, action], np.max(Q_old, axis = 1)) \
                    for action in range(self.env.nA)] \
                        for state in range(self.env.nS)])
            if np.sum(np.abs(Q_old - Q_new)) < delta:
                return Q_new
            Q_old = Q_new
        

    ### ''' Private Updater Methods ''' ###


    def __update_runtime(self, run, step):
        '''
        Update runtime for current step.

        '''
        self.runtime[run, step] = self.get_runtime(run)

    def __update_cumulative_reward(self, run, step, reward):
        '''
        Update cumulative reward.

        '''
        if step == 0:
            self.cumulative_rewards[run, step] = reward
        else:
            self.cumulative_rewards[run, step] = reward + self.cumulative_rewards[run, step-1]


    def __update_std_reward(self, run, state, action, reward):
        '''
        Update standard deviation of reward.
        '''
        self.rewards_history[run, state, action].append(reward)
        if len(self.rewards_history[run, state, action]) > 1:
            self.std_reward[run, state, action] = np.std(self.rewards_history[run, state, action], ddof=1)
        else:
            self.std_reward[run, state, action] = 0.

    
    def __update_coverage_error_squared(self, run, step):
        '''
        Update discrete coverage error.
        Returns the squared error of 

        '''
        self.coverage_error_squared_T[run, step] = np.sum(np.square(self.agent.T - self.env_T))
        self.coverage_error_squared_R[run, step] = np.sum(np.square(self.agent.R - self.env_mean_reward))


    def __update_KL_divergence(self, run, step, state, action):
        '''
        Updates KL divergence metric for transitions

        for T: http://www.java2s.com/Code/Java/Development-Class/ReturnstheKLdivergenceKp1p2.htm
        for R: https://stats.stackexchange.com/questions/234757/how-to-use-kullback-leibler-divergence-if-mean-and-standard-deviation-of-of-two

        '''
        pass
        # with np.errstate(divide='ignore'):
        #     pass
        #     # KL divergence transition: sum(Yt * log(Xt/Yt)) where Yt != 0
        #     agent_transitions = self.agent.T[state, action]
        #     env_transitions = self.env_T[state, action]
        #     klDiv = 0.0
        #     for i in range(len(agent_transitions)):
        #         if agent_transitions[i] == 0 or env_transitions[i] == 0:
        #             continue
        #         else:
        #             klDiv += agent_transitions[i] * np.log( agent_transitions[i] / env_transitions[i])

        #     self.KL_divergence_T[run, state, action] = klDiv

        #     # KL divergence reward: log(std(Xt) / std(Yt)) + (std(Rt)^2 + (mean(Rt) - mean(Yt))^2) / (2 * std(Yt)^2)) - 0.5 where std(Yt) != 0
        #     if self.env_std_reward[state, action] > 0: # and self.std_reward[state][action] > 0
        #         self.KL_divergence_R[run, state, action] = np.log(self.env_std_reward[state, action] / self.std_reward[run, state, action]) + \
        #             (np.square(self.std_reward[run, state, action]) + np.square(self.agent.R[state, action] - self.env_mean_reward[state, action])) / \
        #                 (2 * np.square(self.env_std_reward[state, action])) - 0.5

        # if np.isinf(np.sum(self.KL_divergence_R[run])):
        #     self.KL_divergence_R_sum[run, step] = 2**30
        # else:
        #     self.KL_divergence_R_sum[run, step] = np.sum(self.KL_divergence_R[run])

        # self.KL_divergence_T_sum[run, step] = np.sum(self.KL_divergence_T[run])
        # print(self.KL_divergence_T_sum[run, step])

    def __update_max_policy_agent(self, run):
        '''
        Update max policy of agent

        '''
        self.agent_max_policy[run] = self.agent.max_policy()

    def __update_reward_timeline(self, run, reward, step, state):
        '''
        Update reward timeline for instanteneous loss, all rewards recieved in one array

        '''
        self.reward_timeline[run, step] = reward
        self.state_timeline[run, step] = state


def write_metrics_to_file(list_of_metric_objects, directory, prefix=''):
    '''
    Write metrics to .dat file for vizualisation.
    list_of_metric_objects : list of metric objects, usually one per agent
    directory : Name of output directory
    prefix : prefix of all output files

    '''
    # Variable name : Headers           Var name must be exact match excl. 'self.'
    # First header is the index, others will be prefixed with agent name
    mean_metrics = {
        'runtime' : ['episode', 'runtime'],
        'cumulative_rewards' : ['episode', 'reward'],
        'reward_timeline' : ['episode', 'reward'],
        # 'KL_divergence_T_sum' : ['episode', 'KL_div_T_sum'],
        # 'KL_divergence_R_sum' : ['episode', 'KL_div_R_sum'],
        'coverage_error_squared_T' : ['episode', 'cov_err_sq_T'],
        'coverage_error_squared_R' : ['episode', 'cov_err_sq_R'],
        'instantaneous_loss' : ['episode', 'inst_loss']
    }

    single_metrics = {
        'sample_complexity' : ['run', 'steps']
    }

    for metric in mean_metrics.keys():
        if not hasattr(list_of_metric_objects[0], metric):
            raise AttributeError(f"Metric '{metric}' is not an attribute")

    for metric in single_metrics.keys():
        if not hasattr(list_of_metric_objects[0], metric):
            raise AttributeError(f"Metric '{metric}' is not an attribute")

    BASE_PATH = pathlib.Path(__file__).parent.absolute()
    os.chdir(BASE_PATH)
    if not os.path.exists(directory):
        os.mkdir(directory)

    for metric, headers in mean_metrics.items():

        os.chdir(BASE_PATH)
        os.chdir(directory)

        filename = f'{metric}.dat' if prefix == '' else f'{prefix}_{metric}.dat'

        if os.path.exists(filename):
            os.remove(filename)

        header = headers[0] + '\t' + '\t'.join(f'{obj.name}_{headers[1]}_mean\t{obj.name}_{headers[1]}_std_up\t{obj.name}_{headers[1]}_std_down' for obj in list_of_metric_objects)

        with open(filename, "w") as f:
            f.write(header + '\n')
            for i in range(len(getattr(list_of_metric_objects[0], metric)[0])):
                data = '\t\t'.join(f'{round(np.mean(getattr(obj, metric), axis=0)[i], 5)}\t\t' + \
                    f'{round(np.mean(getattr(obj, metric), axis=0)[i] + np.std(getattr(obj, metric), axis=0)[i], 5)}\t\t' + \
                        f'{round(np.mean(getattr(obj, metric), axis=0)[i] - np.std(getattr(obj, metric), axis=0)[i], 5)}' \
                            for obj in list_of_metric_objects)
                f.write(f'{i+1}\t\t{data}\n')
        f.close()

    for metric, headers in single_metrics.items():

        os.chdir(BASE_PATH)
        os.chdir(directory)

        filename = f'{metric}.dat' if prefix == '' else f'{prefix}_{metric}.dat'

        if os.path.exists(filename):
            os.remove(filename)

        header = headers[0] + '\t' + '\t'.join(f'{obj.name}_{headers[1]}' for obj in list_of_metric_objects)

        with open(filename, "w") as f:
            f.write(header + '\n')
            for i in range(len(getattr(list_of_metric_objects[0], metric))):
                data = '\t\t'.join(f'{round(getattr(obj, metric)[i], 5)}' \
                    for obj in list_of_metric_objects)
                f.write(f'{i+1}\t\t{data}\n')
        f.close()

    
    print('done writing')