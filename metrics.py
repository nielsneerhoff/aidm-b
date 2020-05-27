import numpy as np
import time

class Metrics:
    '''
    Tasked for registering performance of the agents.

    '''

    def __init__(self, agent):

        # Environment
        self.env = agent.env

        # Agent
        self.agent = agent

        # Start Timer
        self.start_time = time.time()

        # Cumulative reward received
        self.cumulative_reward = 0.

        # Transition Function Environment
        self.env_T = self.get_transition_function()

        # Reward distributions
        self.env_mean_reward = np.zeros((self.env.nS, self.env.nA))
        self.env_std_reward = np.zeros((self.env.nS, self.env.nA))
        self.init_reward_distributions()

        self.rewards_history = np.frompyfunc(list, 0, 1)(np.empty((self.env.nS, self.env.nA), dtype=object))
        self.std_reward = np.zeros((self.env.nS, self.env.nA))


        # KL Convergence
        self.KL_divergence_T = np.zeros((self.env.nS, self.env.nA))
        self.KL_divergence_R = np.zeros((self.env.nS, self.env.nA))
        self.init_KL_divergence()

        # Max policy env
        self.env_max_policy = self.get_env_max_policy()

        # Max policy agent
        self.agent_max_policy = self.agent.max_policy()

        # Coverage error transition probabilities and expected reward
        self.coverage_error_squared_T = 0.
        self.coverage_error_squared_R = 0.

        # Sample Complexity Metric
        self.sample_complexity = 0
        self.hit_zero_sample_complexity = False
        self.zero_sample_complexity_steps = -1

        # # Exploration metric
        # self.exploration = np.zeros((self.env.nS, self.env.nA))

        # # Exploitation metric
        # self.exploitation = np.zeros((self.env.nS, self.env.nA))

        # Coverage error
        # self.coverage_error_update_function = {
        #     'discrete' : self.__update_coverage_error_squared,
        #     'uniform' : self.__update_coverage_error_distribution,
        #     'distribution' : self.__update_coverage_error_uniform
        # }[distribution]


    def __str__(self):
        '''
        String representation of the object
        Used for the print() function
        '''
        return f'''
Timer: {self.get_runtime()}
Cumulative reward: {self.cumulative_reward}
Max policy: {self.agent_max_policy}
KL divergence transition distribution:\n{self.KL_divergence_T}
KL divergence reward distribution:\n{self.KL_divergence_R}
Coverage error transition squared: {self.coverage_error_squared_T}
Coverage error reward squared: {self.coverage_error_squared_R}
Sample complexity: {self.sample_complexity}
Hit zero sample complexity after {self.zero_sample_complexity_steps} steps'''

    def get_transition_function(self):
        '''
        Recreate transition function.

        '''
        T = np.zeros((self.env.nS, self.env.nA, self.env.nS))
        for state, actions in self.env.P.items():
            for action, probs in actions.items():
                for (prob, new_state, reward, _) in probs:
                    T[state][action][new_state] = prob
        return T


    def init_reward_distributions(self):
        '''
        Get reward distributions

        '''
        for state, actions in self.env.P.items():
            for action, probs in actions.items():
                self.env_mean_reward[state][action] = sum(reward * prob for (prob, _, reward, _) in probs)
                self.env_std_reward[state][action] = np.sqrt(sum(
                    abs(reward - self.env_mean_reward[state][action])**2 * prob 
                        for (prob, _, reward, _) in probs))

        
    def init_KL_divergence(self):
        '''
        Initialize KL divergence arrays with values.
        
        '''
        for state in range(self.env.nS):
            for action in range(self.env.nA):
                self.__update_KL_divergence(state, action)


    def get_env_max_policy(self):
        '''
        Get optimal policy of the environment

        '''
        return np.argmax(self.value_iteration_env(), axis = 1)



    def value_iteration_env(self, delta = 1e-3):
        '''
        Value iterate over the environment until converged

        '''
        Q_old = np.ones((self.env.nS, self.env.nA))
        while True:
            Q_new = np.array([[self.env_mean_reward[state][action] + \
                self.agent.gamma * np.dot(self.env_T[state][action], np.max(Q_old, axis = 1)) \
                    for action in range(self.env.nA)] \
                        for state in range(self.env.nS)])
            if np.sum(np.abs(Q_old - Q_new)) < delta:
                return Q_new
            Q_old = Q_new
        

    def get_runtime(self):
        '''
        Return runtime up until now.

        '''
        return int(time.time() - self.start_time)

    
    def update_metrics(self, state, action, reward, step):
        '''
        Update from experience.

        '''
        # updates cumulative reward
        self.__update_cumulative_reward(reward)
        
        # updates history of rewards and standard deviation of rewards
        self.__update_std_reward(state, action, reward)
        
        # updates reward and transition error squared
        self.__update_coverage_error_squared()

        # updates KL divergence metric for transitions and rewards
        self.__update_KL_divergence(state, action)

        # updates max policy of agent
        self.__update_max_policy_agent()

        # updates sample complexity, how far from the optimal policy
        self.__update_sample_complexity(step)
        


    def __update_cumulative_reward(self, reward):
        '''
        Update cumulative reward.

        '''
        self.cumulative_reward += reward


    def __update_std_reward(self, state, action, reward):
        '''
        Update standard deviation of reward.
        '''
        self.rewards_history[state][action].append(reward)
        if len(self.rewards_history[state][action]) > 1:
            self.std_reward[state][action] = np.std(self.rewards_history[state][action], ddof=1)
        else:
            self.std_reward[state][action] = 0.

    
    def __update_coverage_error_squared(self):
        '''
        Update discrete coverage error.
        Returns the squared error of 

        '''
        self.coverage_error_squared_T = np.sum(np.square(self.agent.T - self.env_T))
        self.coverage_error_squared_R = np.sum(np.square(self.agent.R - self.env_mean_reward))


    def __update_KL_divergence(self, state, action):
        '''
        Updates KL divergence metric for transitions

        for T: https://stats.stackexchange.com/questions/60619/how-to-calculate-kullback-leibler-divergence-distance?rq=1
        for R: https://stats.stackexchange.com/questions/234757/how-to-use-kullback-leibler-divergence-if-mean-and-standard-deviation-of-of-two

        '''
        # KL divergence transition: sum(Yt * log(Xt/Yt)) where Yt != 0
        agent_transitions = self.agent.T[state][action][self.env_T[state][action] != 0]
        env_transitions = self.env_T[state][action][self.env_T[state][action] != 0]

        self.KL_divergence_T[state][action] = np.sum(env_transitions * np.log(agent_transitions / env_transitions))

        # KL divergence reward: log(std(Xt) / std(Yt)) + (std(Rt)^2 + (mean(Rt) - mean(Yt))^2) / (2 * std(Yt)^2)) - 0.5 where std(Yt) != 0
        if self.env_std_reward[state][action] > 0: # and self.std_reward[state][action] > 0
            with np.errstate(divide='ignore'):
                self.KL_divergence_R[state][action] = np.log(self.env_std_reward[state][action] / self.std_reward[state][action]) + \
                    (np.square(self.std_reward[state][action]) + np.square(self.agent.R[state][action] - self.env_mean_reward[state][action])) / \
                        (2 * np.square(self.env_std_reward[state][action])) - 0.5


    def __update_max_policy_agent(self):
        '''
        Update max policy of agent

        '''
        self.agent_max_policy = self.agent.max_policy()


    def __update_sample_complexity(self, step):
        '''
        Update sample complexity
        Computed as the count of differences between best policy and current policy -> 0 is best

        '''
        self.sample_complexity = np.count_nonzero(self.agent_max_policy != self.env_max_policy)
        if not self.hit_zero_sample_complexity and self.sample_complexity == 0:
            self.hit_zero_sample_complexity = True
            self.zero_sample_complexity_steps = step



    # def __update_coverage_error_distribution(self):
    #     '''
    #     Update coverage error distribution

    #     '''
    #     pass


    # def __update_exploration(self, reward):
    #     '''
    #     Update cumulative reward

    #     '''
    #     self.cumulative_reward += reward


    # def __update_exploitation(self, reward):
    #     '''
    #     Update exploitation

    #     '''
    #     self.cumulative_reward += reward


    # def __update_coverage_error_uniform(self):
    #     '''
    #     Update coverage error uniform

    #     '''
    #     pass


