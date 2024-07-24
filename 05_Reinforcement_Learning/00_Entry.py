# Reinforcement Learning is a powerful branch of Machine Learning.
# It is used to solve interacting problems where the data observerd up to time "t" is considered to decide which action to take at time "t + 1".
# it is also used for Artificial Intelligence when training machines to perform tasks such as walking.
# Desired outcomes provide the AI with reward, undesired with punishment. Machines learn through trial and error.
# In this part we will understand and learn how to implement the following models:

# 1. Upper Confidence Bound (UCB)
# 2. Thompson Sampling

#Examples of Reinforcement learning:

# The Multi-Armed Bandit Problem

# explanation: a one-armed bandit is a slot machine (from the past) where instead having an armed that triggered the game, now we have buttons. It was called bandit as the chances of winning were really low.
# a multi armed bandit problem is kind of the challange that the person is faced when he comes up with a hole set of these machines. The question is how do you play them in order to maximise your return on investment. The assumption is that each one of these machines has a different distribution behind it. So we don't know these chances of winning of each machine. We need to figure it out which distribution is best for us :)
# the longer time we spend on these machine, the more money we loose. So we need to be quick.
# but also, if we don't explore enough time we might not be able to take the right decision.
# 2 factors: exploration and expectation
# "Regret" is something that we can quantify for each machine



# Campaign by Coca-Cola called: Welcome to the Coke Side of Life"

# and we have 10 images / ads that we want to find out which one is the best one, that gives us the best campaign results.
# So there is no known distribution for these ads, unless thousands of ad clicks, which is not the case.
# One approach is to do a A/B testing, take aprox 10% of the ads and run a huge A/B test. The problem is that we need to wait a long time and we loose too much money.
# but A/B tests are just for exploration.
# But at the beginning of the exploration we can figure it out which compaign to start with to minimise the time and money spent.
# this example will be used in our coding sample


# the multi-armed bandit problem

# * We have d arms. For example, arms are ads that we display to users each time they connect to a web page
# * Each time a user connects to this web page, that makes a round
# * At each round n, we choose one ad to display to the user
# * At each round "n", ad "i" gives "Reward" Ri belongs to {0,1}: Ri = 1 if the user clicked on the add "i", 0 if the user didn't
# * Our goal is to maximize the total reward we get over many rounds.
 
# Steps:
# Step 1: At each round "n", we consider 2 numbers for each ad "i":
#   Ni(n) - the numbers of times the ad "i" was selected up to round "n",
#   Ri(n) - the sum of rewards of the ad "i" up to round "n".

# Step 2: From these 2 numbers we computer:
#   * the average reward of ad "i" up to round "n" = Ri(n) / Ni(n)
#   * the confidence interval [ ri(n) - ( delta i (n), r i (n) ) + delta i (n) ] at round n with delta i (n) = Squared root of 3/2 * log(n) / Ni(n)

# Step 3: We welect the ad "i" that has the maximum UCB: r i (n) + delta i (n)
