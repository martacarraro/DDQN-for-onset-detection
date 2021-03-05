import gym
import gym_signal

import random
import numpy as np
from collections import deque

import keras
from keras import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input


import matplotlib
import matplotlib.pyplot as plt

import os




class ReplayBuffer:
    def __init__(self, maxlen = 4000):
        self.buffer = deque(maxlen=maxlen)


    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        #print("length: ", len(self.buffer))

    def get_batch(self, batch_size):
        if batch_size > len(self.buffer):
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

    def get_size(self):
        return len(self.buffer)



GAMMA = 0.9
#ALPHA = 0.5
EPSILON = 1.0
EPSILON_MIN = 0.1
LEARNING_RATE = 0.001


class DDQN:

    def __init__(self, state_space, action_space, n_episodes):
        """
        Arguments: - state_size: size of the state
        """

        # Initialize attributes
        self.state_size = state_space # TODO: FIX THIS||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
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
        self.q_network.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

        # target network
        self.target_network = self.build_network('target_network')

        # copy q-network weights to target network------------------------------
        self.counter = 0
        self.update_target_model()#,metrics


    def buffer_size(self):
        return self.buffer.get_size()

    def make_interval(self,state):
        state_interval = list(range((state[0]-2), (state[0]+2+1)))
        #(remove from the intervals all the values < 0 and >= max length of the signal)
        state_interval = [item for item in state_interval if (item >= 0) & (item < self.state_size.n)]
        if len(state_interval)==3:
            state_interval.insert(0,0)
            state_interval.insert(4,0)
        if len(state_interval)==4:
            state_interval.insert(0,0)
        return np.array(state_interval).reshape(1,-1)

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
            state_interval = self.make_interval(state)

            q_values = self.q_network.predict(state_interval)
            action = np.argmax(q_values[0])
            return action





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
        input_size = 5
        output_size = self.action_space.n

        net = Sequential(name=network_name)
        net.add(Input(shape=(input_size,)))
        net.add(Dense(8, activation='relu'))
        net.add(Dense(8, activation='relu'))
        net.add(Dense(output_size, activation='softmax'))

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

        """
        state is a tuple (position, d), giving these 2 informations I create an
        interval [position-d,position+d] that will be the input of the network. The
        network accepts 5 nodes in input...if d=1 I zero-pad the interval in order
        to reach dimension (5,)....then I transform this list to a numpy array with
        shape (1,5)
        """

'''
    def get_target_q_value(self, next_state, reward):
        """ it computes the target q_value
        Arguments:  - next_state: next state
                    - reward: reward obtained after making action from state
        Returs: - q_value: target q-value
        """
        next_state_interval = self.make_interval(next_state)

        print("      next state interval: ", next_state_interval)

        # q_network selects the action a'_max = argmax_a' Q(s',a')
        action = np.argmax(self.q_network.predict(next_state_interval)[0])
        print("      (qnet) next state q values: ", self.q_network.predict(next_state_interval)[0])
        print("      choosed action:", action)
        # target network evaluated the q value Q_max = Q_target(s',a'_max)
        q_value = self.target_network.predict(next_state_interval)[0][action]
        print("      (tnet) next state q values: ", self.target_network.predict(next_state_interval)[0])
        print("      q_value: ", q_value)
        # return the qvalue Q_max = reward + gamma*Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

'''

    def replay(self, batch_size):

        # get a batch of trajectories from replay buffer
        sars_batch = self.buffer.get_batch(batch_size)
        state_batch = []
        q_values_batch = []

        print("start for cycle in BATCH of size: ", batch_size)
        i=0
        for state,action,reward,next_state,done in sars_batch:
            # create input for q_network
            print(i)

            print("state: ", state)
            print("action: ", action)
            print("next state: ", next_state)
            print("reward: ", reward)
            print("done:", done)
            state_interval = self.make_interval(state)
            next_state_interval = self.make_interval(next_state)

            '''
            # prediction on a certain state
            q_values = self.q_network.predict(state_interval) #--> ouput layer given a state in input of the q_network
            print("q_values: ", q_values)
            # get target q_value
            print("call get_target_q_value(next_state, reward):.....")
            q_value = self.get_target_q_value(next_state,reward)

            print("updated q_value (with bellman): ", q_value)
            #correction on the q value for the action used
            q_values[0][action] = reward if done else q_value
            print("predicted q_value:", q_values[0][action])


            # collect batch state-q-value mapping
            state_batch.append(state_interval[0])
            q_values_batch.append(q_values[0])
            '''
            target = self.q_network.predict(state_interval)
            if done:
                target[0][action] = reward
            else:
                next_action = np.argmax(self.q_network.predict(next_state_interval)[0])
                target[0][action] = reward + self.gamma * self.target_network.predict(next_state_interval)[0][next_action]

            # collect batch state-q-value mapping
            state_batch.append(state_interval[0])
            target_batch.append(target[0])

            i+=1

        # perform one step of gradient descent
        self.q_network.fit(np.array(state_batch), np.array(target_batch), epochs=1, verbose=1)





        # train the q-network
        #self.q_network.fit(np.array(state_batch), np.array(q_values_batch), batch_size=batch_size, epochs=1, verbose=1)


        # update explotation-exploitation rate
        self.update_epsilon()

        # update the target network parameters every C training steps
        C = 10
        if self.counter % C == 0:
            self.update_target_model()
            print('UPDATE TARGET NETWORK PARAMETERS....')
        self.counter += 1



    def save(self, folder_name):

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.q_network.save(folder_name + '/q_net.h5')
        self.target_network.save(folder_name + '/target_q_net.h5')
        # Save replay buffer
        self.buffer.save(folder_name + '/replay-buffer')

        #json?
        # Save meta
        #with open(folder_name + '/meta.json', 'w+') as f:
        #    f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))  # save replay_buffer information and any other information


    def load(self, folder_name, load_buffer = True):
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load networks
        self.q_network = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_network = tf.keras.models.load_model(folder_name + '/target_dqn.h5')

        # Load replay buffer
        if load_buffer:
            self.buffer.load(folder_name + '/replay-buffer')








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


"""
v = action_space.sample()
print()
print(f'Space: {action_space}')
print(f'Sample: {v}')
print(f'Flatdim = {gym.spaces.flatdim(action_space)}')
print(f'Flatten = {gym.spaces.flatten(action_space, v)}')
"""
