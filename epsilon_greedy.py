"""
We solve the multi arm bandit problem by epsilon greedy approach.
"""

"""
Algorithm:
1. Choose an epsilon(0.1), the epsilon usually defines the error rate, suppose we want our agents to
fulfill a task, if the error rate is less than the defined(0.1) we can choose any agent.
2. If the error rate is more we can go back to exploration, in exploration we choose arm with
best possible success rate or reward.
"""



"""
	1.Epsilon is any float number
	2.Counts is the number of times each arm/agent has been called, so if there are two agents
	and each agent has been called 2 and 3 times respectively it would be [2,3].
	3. Values represent the success rate or rewards that the agent/arm have recieved, [0.3,0.4]
	for two agents.
	"""

import random
import numpy as np

def ind_max(x):
  m = max(x)
  return x.index(m)

class EpsilonGreedy():
  def __init__(self, epsilon, counts, values):
    self.epsilon = epsilon
    self.counts = counts
    self.values = values


  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    if random.random() > self.epsilon:
      return ind_max(self.values)
    else:
      return random.randrange(len(self.values))
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return self.counts, self.values

