'''
Classes that compute:
    * P(i) = Probability of an item
    * P(t) = Probability of a tag
    * P(u) = Probability of an user
    * P(u|i) = Probability of an user given an item
    * P(t|i) = Probability of a tag given an item
'''

from .ProbabilityEstimator import ProbabilityEstimator
from .RWEstimator import RWEstimator
from .SmoothEstimator import SmoothEstimator