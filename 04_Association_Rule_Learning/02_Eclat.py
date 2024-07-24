# Eclat model: simplified version of Apriori -> it tells us what other items the customers are likely to add into the basket, this is a SET, is like one item goes hand in hand with another, or a SF movie is bought together with another SF movie

# in the Eclat model we only have support factor and it is much faster

# support factor (M) = [# user watchlists containing M] / [# user watchlists]

# Step 1: Set a minimum support
# Step 2: Take all the subsets in transactions having higher support than minimum support
# Step 3: Sort these subsets by decreseasing support.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv(r"04_Association_Rule_Learning\Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
# print(results)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
# print(resultsinDataFrame)

## Displaying the results sorted by descending supports
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))
