"""
The objectrive for the agent is to reach the cake, as -----C where 'C' is the cake,
The environment is "-----C" with each "-" representing a state and from each state
we either go "left" or "right", since the destination lies on right, "C", the agent 
will get a + reward if it goes left and not rewarded if it goes right.
Q(s,a) = r + gamma*(max(Q(s',a')))
"""

import numpy as np 
import random
import sys
import pandas as pd
np.random.seed(2)

num_states = 6
actions = ["left","right"]
alpha = 0.1 #learning_rate
gamma = 0.9
max_eps = 13
fresh_time = 0.3 #fresh time for one move 

actions = ["left","right"]

#build a q-table

"""
The q-table will contain the dimensions of num_states x len(actions), for each state there will be a left or right action to be performed.
"""
q_table = np.zeros((num_states,len(actions)))
q_table = pd.DataFrame(q_table,columns=['left','right'])

#choose a state:

"""
choose a state, there are 6 states altogether, 0,1,2,3,4,5. 5 being the destination state:
1. Choose randomly a state
2. If the state is 5, then we have reached the destination, we have not.
"""

cur_position = random.choice(range(6))
if cur_position==num_states-1:
	cur_pos='Terminal'
	print ("Already reached destination")
	sys.exit(1)
else:
	cur_pos = cur_position

#choose a action:

"""
Now the action will be selected based upon two conditions: either you have a previous score, if you have then select the action based upon that else, randomly select the action.
The cur_pos will describe your state, from there by selecting the max value we can,
"""

def choose_action(q_table,cur_pos):
	state_actions = q_table.loc[cur_pos,:]
	if (state_actions==0.0).all():
		choose_action = np.random.choice(actions)
		print (choose_action)
	else:
		choose_action = state_actions.idxmax()
	return choose_action

def get_reward(action_,state):
	if action_=='right':
		if state == num_states-1:
			new_state = "terminal"
			r=1
		else:
			new_state = state+1
			r=0
	else:
		r=0
		if state ==0:
			print ("Reached The Wall")
			new_state = state
		else:
			new_state = state -1 
	return new_state, r

def rl():
	for eps in range(13):
		print ("*********************EPISODE****************",eps)
		reach_terminal = False

		state =0

		while not reach_terminal:
			action_ = choose_action(q_table,state)
			#For the action chosen get reward:
			new_state, reward = get_reward(action_,state)
			q_predict = q_table.loc[state, action_]
			if new_state!='terminal':
				q_target = reward + gamma*q_table.iloc[new_state,:].max()
			else:
				q_target = reward
				reach_terminal = True

			q_table.loc[state, action_] += alpha * (q_target - q_predict)

			state= new_state

		print (q_table)

if __name__ == '__main__':
	rl()
















