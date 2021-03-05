# to use RL env
import gym
import gym_signal

import random
import numpy as np
from collections import deque

# to build NNs
import keras
from keras import Sequential
from tensorflow.keras.layers import Dense, Input

# to make plots
import matplotlib
import matplotlib.pyplot as plt


# to use the timer
import timeit
#from timeit import default_timer

# to use DDQN objects
from DDQN import *


import os






N_EPISODES = 100
BATCH_SIZE = 32
log_name = 'DDQN/check_time'
ACTIONS = ['LEFT', 'STAY','RIGHT','TWO STEPS RIGHT','d=1','d=2']


# create the signal
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



	log_file = open(log_name, 'a')
	log_file.write('\nSTART MAIN:\n')
	log_file.close()

	[x, signal] = generate_signal()
	env = gym.make('Signal-v0')
	state_space = env.observation_space
	action_space = env.action_space
	print("Observation space:", state_space)
	print("Action space:", action_space)

	ddqn_agent = DDQN(state_space, action_space, N_EPISODES)

	simulation_time = 0
	cycle_time = 0
	scores = []
	epsilon_values = [1]
	for episode in range(N_EPISODES):
		start = timeit.default_timer()

		if episode % 1 == 0:
			log_file = open(log_name, 'a')
			log_file.write('Episode '+ str(episode)+ ' --- time: '+ str(cycle_time) + ' s')
			log_file.write('\n')
			log_file.close()


		#reset environment at the beginning of every episode
		state = env.reset()

		print("----------------------------START episode:", episode,'------------------------------')
		print("initial state: ", state)
		total_reward = 0
		done = False
		i=0
		while not done:
			# choose an action
			action = ddqn_agent.act(state)
			print("\nchoose action: ", action, ' (', ACTIONS[action],')')

			# make that action and observe reward and next state
			next_state, reward, done, _ = env.step(action)

			# save the trajectory in the memory buffer
			ddqn_agent.store_trajectory(state, action, reward, next_state, done)
			print("after calling step function...\nstate:", state, "---->  next state:", next_state) #next state
			print("reward:",reward)
			print("done:",done)

			# the new state becomes the current state
			state = next_state
			# accumulate reward for a single episode
			total_reward += reward

			print("\ntotal reward: ", total_reward)
			print("buffer size: ", ddqn_agent.buffer.get_size())
			print("iteration ", i, "of episode", episode)
			i+=1




			if ddqn_agent.buffer.get_size() >= BATCH_SIZE:
				print("\n\nTRAIN NETWORK------------------------------------------------")
				ddqn_agent.replay(BATCH_SIZE)
				scores.append(total_reward)
				epsilon_values.append(ddqn_agent.epsilon)
				print("epsilon:", ddqn_agent.epsilon)

			if done:
				break

		# end of the episode
		stop = timeit.default_timer()

		cycle_time = (stop - start)
		simulation_time += cycle_time

		if episode % 10 == 0:
			elapsed_time = simulation_time/60
			log_file = open(log_name, 'a')
			log_file.write('...elapsed time: '+ str(elapsed_time) + ' min')
			log_file.write('\n')
			log_file.close()
		#simulation_time += (stop - start) / 60
		#cycle_time = (N_EPISODES - episode - 1)*simulation_time/(episode+1)


		# print current scores -------------------------------------------------------
		if episode % 10 == 0:
			tt = np.linspace(0, len(scores), len(scores))
			plt.figure()
			plt.plot(tt,scores)
			plt.xlabel('episodes')
			plt.ylabel('rewards')
			plt.xticks(range(0, len(scores)))
			plt.grid()
			plt.savefig('DDQN/plots5/scores_' + str(episode) +'.png')




	print("scores: ", scores)
	simulation_time = simulation_time/60 #min
	print("total time:", simulation_time, " min")

	# print epsilon decay ------------------------------------------------------
	t1 = np.linspace(0, len(epsilon_values), len(epsilon_values))
	plt.figure()
	plt.plot(t1,epsilon_values)
	plt.xlabel('episodes')
	plt.ylabel('epsilon')
	plt.ylim(0,1)
	plt.xticks(range(0,len(epsilon_values)))
	plt.grid()
	plt.savefig('DDQN/plots5/epsilon_decay.png')

	# print total scores -------------------------------------------------------
	t2 = np.linspace(0, len(scores), len(scores))
	plt.figure()
	plt.plot(t2,scores)#,label='rewards')
	plt.xlabel('episodes')
	plt.ylabel('rewards')
	plt.xticks(range(0, len(scores)))
	plt.grid()
	plt.savefig('DDQN/plots5/total_scores.png')
