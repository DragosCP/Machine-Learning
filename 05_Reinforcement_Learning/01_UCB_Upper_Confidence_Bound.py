

# We're going to find the best AD among different ad designs (10 in total) that will maximise the nr of customers that click on the ad, potentially to buy the SUV


# our db is called ads_CTR_optimisation : CTR means click to rate
# this dataset is a simmulation!!

# we're going to show each of the 10 Ads to 10,000 users and record the inputs, CRT yes or no, 0 or 1

# each AD has a fixed conversion rate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"05_Reinforcement_Learning\Ads_CTR_Optimisation.csv")

# implementing UCB
import math
N = 10000 # times (rounds) a user clicked on the add
d = 10 # number of ads
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0

    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # we do log n+1 because log 0 = infinit / or we can leave log(n) and start the for loop from (1, d)
            upper_bound = average_reward + delta_i
        else: 
            upper_bound = 1e400 # infinite
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward



# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# the result with N = 10000 and 5000 is clear, AD no. 4 wins
# the result with N = 1000 is on the edge, AD no. 4 also wins
# the result with N = 500 is giving us that AD no 7 is winning.

