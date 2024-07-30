# Thompson Sampling Algorithm

# We have d arms. For example, arms are ads that we display to users each time they connect to a web page
# Each time a user connect to this web page, that makes a round
# At each round n, we choose one ad to display to the user
# At each round n, ad i give reward r i (n) belongs to {0, 1}: r i (n) = 1 if the user clicked on the ad i, 0 if didn't
# Our goal is to maximize the total reward we get over many rounds.

# Bayesian Inference

# The Thompson Sampling Algorithm is a probabilistic algorithm: it has distributions which represents our perception of the world and where we think the actual expected returns of each of the machines might lie, so we actually generate random values from those distributions. 
# 
# In the UCB, after we've received the previous value from the machine, and then we re-run the round, it's always going to be the same result,
# whereas in the Phompsons, after we receive the previous value from the machine and we re-run the current round, it's always going to be different result because we're always sampling from our distributions which caracterize our perception of the world.

# implications:
# UCB requires an update every round, so we store the value after the round and make adjustments to the alg based on that new value
# Thompsons can accomodate delayed feedback, if we pull the lever 500 rounds down the track (not right away) we will get to know the results 500 rounds later, and will still work. Because if we now run the alg without even updating our perception of the world, we're still going to get a new set of hypothetical bandits. We're going to generate a new expected return for every bandit because we are generating them in a probabilistic manner.

# This is very important in the ADs space as we could update the algorithm in a batches of clicks, let's say 500-5000. So we can get updated results often.

# finding the AD with the highest converting rate.

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"05_Reinforcement_Learning\Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling
import random
N = 1000 # no of users/clicks
d = 10 # no of ads
ads_selected = [] # will be populated with different ads selected at each round
numbers_of_rewards_1 = [0] * d # the number of times the ad 'i' got reward 1 up to round 'n'
numbers_of_rewards_0 = [0] * d # ---------------------------------- reward 0 -------------
total_reward = 0 # accumulated rewards over time
for n in range(0, N):
    ad = 0 # the ad that we select at each round n
    max_random = 0 # max random draws that we will compare with the highest draw (in step 3)
    for i in range(0, d): # it gives us the ad that has the max random draw among all the ads from 0 to 9
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) #random draw from beta distribution of our parameters
        if random_beta > max_random: #step 3
            max_random = random_beta
            ad = i
        # else we will just keep the max random draw - so don't need to do anything
    ads_selected.append(ad)
    reward = dataset.values[n, ad] # the value in the dataset corresponding to the row we are dealing with right now in this first full loop, with this particular customer and the column of the AD that was just selected.
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1

    total_reward += reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# in the UCB we couldn't get the same results for 500 rounds as 10,000 rouds
# in the Thompsons we actually get the same result with 500 rounds but also with 10,000 rounds, so in our case is better to use this algorithm