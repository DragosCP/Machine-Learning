# example: Let's assume there is a factory line producing wrenches and a second line producing the same wrenches, but we know which ones come from wich line as they are tagged.
# all the wrenches are now pilled together and the workers at the factory go through them to find out the defective wrenches hiding among the pile.
# the question: what is the probability of each factory Line to producing a defective wrench?

# The Bayes Theorem:

# P(A\B) = [ P(B\A) * P(A)] / P(B)

# P = probability
# 

# Let's assume Line 1 : 30 wrenches / hr ; Line 2: 20 wrenches / hr
# Let's assume all produced parts: 1% are defective, we can see 50% came from Line 1 and 50% Line 2
# Q: what is the probability that a part produced by Line 2 is deffective?

# -> P(line1) = 30/50 = 0.6
# -> P(line2) = 20/50 = 0.4
# -> P(Defect) = 1% 
# -> P(Line1|Defect) = 50%  "|" given some conditions


