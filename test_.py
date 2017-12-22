import random
import numpy as np
from epsilon_greedy import EpsilonGreedy 
from bernoulli_arm import BernoulliArm

def test_algorithm(algo,arms,num_sims,horizon):
	chosen_arms= [0.0 for i in xrange(num_sims*horizon)]
	rewards= [0.0 for i in xrange(num_sims*horizon)]
	cumulative_rewards= [0.0 for i in xrange(num_sims*horizon)]
	sim_nums= [0.0 for i in xrange(num_sims*horizon)]
	times= [0.0 for i in xrange(num_sims)]

	for sim in xrange(num_sims):
		sim= sim+1
		algo.initialize(len(arms))

	for t in xrange(horizon):
		t=t+1
		index_= (sim-1)*horizon +t -1

		sim_nums[index_]= sim
		times[index_]=t 

		chosen_arm= algo.select_arm()
		chosen_arms[index_]= chosen_arm

		reward= arms[chosen_arms[index_]].draw()
		rewards[index_]= reward

		if t==1:
			cumulative_rewards[index_]=reward 
		else:
			cumulative_rewards[index_]= cumulative_rewards[index_ - 1]+reward

		algo.update(chosen_arm,reward)
	return [sim_nums,times,chosen_arms,rewards,cumulative_rewards]


if __name__ == '__main__':
	random.seed(1)
	means= [0.1,0.1,0.1,0.1,0.9]
	n_arms= len(means)
	random.shuffle(means)
	arms= map(lambda (mu): BernoulliArm(mu),means)
	print ("Best Arm is")

	#lets choose an epsilon to test the epsilon_greedy algo:
	eps= 0.1
	algo_= EpsilonGreedy(eps,[],[])
	algo_.initialize(n_arms)
	num_sims= 5000
	horizon= 250
	chosen_arms= [0.0 for i in xrange(num_sims*horizon)]
	print len(chosen_arms)
	rewards= [0.0 for i in xrange(num_sims*horizon)]
	print len(rewards)
	cumulative_rewards= [0.0 for i in xrange(num_sims*horizon)]
	print len(cumulative_rewards)
	sim_nums= [0.0 for i in xrange(num_sims*horizon)]
	times= [0.0 for i in xrange(num_sims*horizon)]

	for sim in xrange(num_sims):
		sim=sim+1
		algo_.initialize(len(arms))

		for t in xrange(horizon):
			t=t+1
			index_= (sim-1)*horizon + t -1

			sim_nums[index_]= sim
			times[index_]= t

			chosen_arm= algo_.select_arm()
			chosen_arms[index_]= chosen_arm

			reward= arms[chosen_arms[index_]].draw()
			rewards[index_]= reward

			if t==1:
				cumulative_rewards[index_]=reward
			else:
				cumulative_rewards[index_]= cumulative_rewards[index_ - 1] +reward

			algo_.update(chosen_arm,reward)






