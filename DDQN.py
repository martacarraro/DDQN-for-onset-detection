import gym
import gym_signal

import random
import numpy as np
from collections import deque

import keras
from keras import Sequential
from tensorflow.keras.layers import Dense, Input


class ReplayBuffer:
    def __init__(self, maxlen = 2000):
        self.buffer = deque(maxlen=maxlen)


    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        #print("length: ", len(self.buffer))

    def get_batch(self, batch_size):
        if batch_size > len(self.buffer):
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)
    """
    def get_arrays_from_batch(self, batch):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([(np.zeros(NUM_STATES) if x[3] is None else x[3])
                                for x in batch])

        return states, actions, rewards, next_states
    """

    def get_size(self):
        return len(self.buffer)



GAMMA = 0.9
ALPHA = 0.5
EPSILON = 1.0
EPSILON_MIN = 0.1


class DDQN:

    def __init__(self, state_space, action_space, n_episodes):
        """
        Arguments: - state_size: size of the state
        """

        # Initialize attributes
        self.state_size = state_space # TODO: FIX THIS|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\\
        self.action_space = action_space

        # discount rate -- gamma determines how much the agent prioritizes current over future rewards.
        self.gamma = GAMMA
        # epsilon rate
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = (self.epsilon_min/self.epsilon)**(1./float(n_episodes))

        # create replay buffer--------------------------------------------------
        self.buffer = ReplayBuffer()

        # build networks--------------------------------------------------------
        # q-network
        self.q_network = self.build_network('q_network')
        self.q_network.compile(optimizer='adam', loss='mse')

        # target network
        self.target_network = self.build_network('target_network')

        # copy q-network weights to target network------------------------------
        self.counter = 0
        self.update_target_model()#,metrics


    def buffer_size(self):
        return self.buffer.get_size()

    def act(self, state):
        """
        Agent chooses the action following the epsilon-greedy policy. This
        function takes the current state of the agent and chooses an action
        based on the current values in the Q-table and the value of epsilon
        (it chooses a random action if a randomly chosen value is less than
        epsilon). NB. use model.predict() to retrieve the Q-values
        """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample() #sample() is a function of spaces.Discrete()
        else:
            q_values = self.q_network.predict(state) # TODO: CHECK WHAT IS THE OUTPUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
            action = np.argmax(q_values[0])
            return action

    def get_target_q_value(self, next_state, reward):
        """ it computes the target q_value
        Arguments:  - next_state: next state
                    - reward: reward obtained after making action from state
        Returs: - q_value: target q-value

        """
        # q_network selects the action a'_max = argmax_a' Q(s',a')
        action = np.argmax(self.q_network.predict(next_state)[0])
        # target network evaluated the q value Q_max = Q_target(s',a'_max)
        q_value = self.target_network.predict(next_state)[0][action]
        # return the qvalue Q_max = reward + gamma*Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value


    def update_target_model(self):
        """
        It copies the weights of the trained q network to the target network
        """
        self.target_network.set_weights(self.q_network.get_weights())
        # vedere se funziona invece questo
        #for t, e in zip(self.target_network.trainable_variables,
        #            self.primary_network.trainable_variables): t.assign(t * (1 - TAU) + e * TAU)


    def build_network(self, network_name):
        """
        It builds the network architecture
        """
        input_size = self.state_size.n
        output_size = self.action_space.n

        net = Sequential(name=network_name)
        net.add(Input(shape=(input_size,)))
        net.add(Dense(8, activation='relu'))
        net.add(Dense(8, activation='relu'))
        net.add(Dense(output_size, activation='linear'))
        # add BatchNormalization, Dropout, kernel initializer? bias initializer?
        net.summary()
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return net

    def update_epsilon(self):
        """
        decrease the explotation, increase the exploitation
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def store_trajectory(self, state, action, reward, next_state, done):
        """
        Store a trajectory (sars) in the replay buffer
        """
        self.buffer.store(state, action, reward, next_state, done)


    def replay(self, batch_size):
        # get a batch of trajectories from replay buffer
        sars_batch = self.buffer.get_batch(batch_size)
        state_batch = []
        q_values_batch = []

        for state,action,reward,next_state,done in sars_batch:
            # create input for q_network
            '''
            print(state)
            print(action)
            print(next_state)
            input = list(range((state[0]-2), (state[0]+2+1)))
            input = np.clip(input, 0, 100 - 1)
            print(input)
            '''
            # prediction on a certain state
            q_values = self.q_network.predict(input) #--> ouput layer given a state in input of the q_network
            # get target q_value
            q_value = self.get_target_q_value(next_state,reward)
            q_values[0][action] = reward if done else q_value

            #collect batch state-q-value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])



        # train the q-network
        self.q_network.fit(np.array(state_batch), np.array(q_values_batch), batch_size=batch_size, epochs=1, verbose=0)


        # update explotation-exploitation rate
        self.update_epsilon()

        # update the target network parameters every C training steps
        C = 10
        if self.counter % C == 0:
            self.update_target_model()
        self.counter += 1







def generate_signal(seed=0):
	x = np.linspace(0, 100, 100) #start value of the sequence, end value of the seqence, number of samples to generate. Default is 50.
	mask1 = (x>=10) & (x<=17)
	mask2 = (x>=28) & (x<=35)
	mask3 = (x>=64) & (x<=77)
	mask4 = (x>=88) & (x<=93)

	y = np.where(mask1, 20, 0) + np.where(mask2, 10, 0) + np.where(mask3, 30, 0) + np.where(mask4, 5, 0)
	np.random.seed(seed)
	noise = np.random.normal(0,0.5,100)

	return [x,y + noise]





if __name__ == "__main__":
    """
    [x, signal] = generate_signal()
    env = gym.make('Signal-v0')
    obs = env.reset()

    state_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", state_space)
    print("Action space:", action_space)

    v = action_space.sample()
    print()
    print(f'Space: {action_space}')
    print(f'Sample: {v}')
    print(f'Flatdim = {gym.spaces.flatdim(action_space)}')
    print(f'Flatten = {gym.spaces.flatten(action_space, v)}')
    """




    [x, signal] = generate_signal()
    env = gym.make('Signal-v0')
    state_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", state_space)
    print("Action space:", action_space)

    n_episodes = 1
    ddqn_agent = DDQN(state_space, action_space, n_episodes)

    for episode in range(n_episodes):

        #reset environment at the beginning of every episode
        state = env.reset()

        print("state: ", state)
        total_reward = 0
        done = False
        i=0
        while not done:
            # choose an action
            action = ddqn_agent.act(state)
            print("action: ", action)

            # make that action and observe reward and next state
            next_state, reward, done, _ = env.step(action)

            # save the trajectory in the memory buffer
            ddqn_agent.store_trajectory(state, action, reward, next_state, done)
            print("next state:", next_state) #next state
            print("reward:",reward)
            print("done:",done)

            # the new state becomes the current state
            state = next_state
            # accumulate reward for a single episode
            total_reward += reward

            #if i == 10: done=True
            i += 1
            print("total reward: ", total_reward)
            print("buffer size: ", ddqn_agent.buffer.get_size())

            ddqn_agent.replay(10)
            print("epsilon:", ddqn_agent.epsilon)
            #print("len buffer: ", ddqn.buffer.buffer_size())
            print('-----------------------------------------')
        print("# iterations of an epoch:", i)






"""
# Q-Learning sampling and fitting
for episode in range(episode_count):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    while not done:
        # in CartPole-v0, action=0 is left and action=1 is right
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        # in CartPole-v0:
        # state = [pos, vel, theta, angular speed]
        next_state = np.reshape(next_state, [1, state_size])
        # store every experience unit in replay buffer
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    # call experience replay
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)
        scores.append(total_reward)
        mean_score = np.mean(scores)
    if mean_score >= win_reward[args.env_id] and episode >= win_trials:
        print("Solved in episode %d: Mean survival = %0.2lf in %d episodes" % (episode, mean_score, win_trials))
        print("Epsilon: ", agent.epsilon)
        agent.save_weights()
        break
    if (episode + 1) % win_trials == 0:
        print("Episode %d: Mean survival = %0.2lf in %d episodes" % ((episode + 1), mean_score, win_trials))
"""
