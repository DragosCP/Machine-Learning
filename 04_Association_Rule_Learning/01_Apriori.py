# Aprori algorithm has 3 parts: (example movies M)
#   Support (M) = [# user watchlists containing movie M ]  / [# all watchlist ]
#   Confidance (M1 -> M2) = [# user watchlists containing M1 and M2 ]  / [#user watchlist containing M1]
#   Lift (M1 -> M2) = (what we had in the Naive Bayes) = [ confidence(I1->I2) ] / Support

# Support: out of all 100 users, how many users did watch "Ex-machina", let's say 10. Support = 10%
# Confidence: Out of 100 people, 40 watched "Interstellar", how many did also watch "Ex machina", let's say 7. Confidence = 7/40 = 17.5%
# Lift : Confidence / Support = 17.5% / 10% = 1.75%

# STEPS:
# 1. Set a minimum support and confidence
# 2. Take all the subsets in transactions having higher support than minimum support
# 3. Take all the rules of these subsets having higher confidence than minimum confidence
# 4. Sort the rules by decreasing lift.

# before we either predicted a dependent varaible and we knew what to predict and also learned patterns in the data with clustering as to create a new dependent variable, a posteriori but now we're going to learn some associati on rules inside an ensemble of transactions.

# We have a db where the owner of the store did register all the items each client bought (in one basket) at once together. The idea is to, after some time, to make some offers with some items that the clients are more likely to add on the basket on top of what they're looking to buy.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"04_Association_Rule_Learning\Market_Basket_Optimisation.csv", header = None) # as we don't have a row with titles like : price, id, name, etc
# Apriori works with lists of strings only. 
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Train the Apriori model on the dataset and returning the rules (support, confidences and lifts)
# the total number of 7500 transaction are over a week
# min support = [ min 3 products purchased per day x 7 days ] / 7501 ~ 0.003
# min confidence = common practice, 0.2 to 0.8 ?
# min lift = should be minimum 3
# min_length / max = the rule must have 2 products at the end, 1 product on the left hand side of the rule and 1 on the right hand side of the rule -> we're specifying an example with buy 1 get 1 free. If we want different deals, like buy 2 product, get one for free, we change these variables to 3
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
# print(resultsinDataFrame)

## Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))