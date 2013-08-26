# -*- coding: utf8
'''
Contains some methods for creating probability estimators with common 
parameters. 
'''
from __future__ import division, print_function

from tagassess.probability_estimates.lda_estimator import LDAEstimator
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator

def create_lda_estimator(annotations_it, gamma, num_items, num_tags, 
        num_topics=200):
    '''
    Creates the lda estimator with the parameters described in [1]_. Alpha and
    Beta are defined as a function of the number of items and tags, thus only
    gamma is needed to be varied. 
    
    References
    ----------
    ..[1] Harvey, M., Ruthven, I., & Carman, M. J. (2011). 
    "Improving social bookmark search using personalised latent variable 
    language models." 
    Proceedings of the fourth ACM international conference on Web search and 
    data mining - WSDM  ’11. doi:10.1145/1935826.1935898
    '''
    
    alpha = 0.1 * len(num_items)
    beta = 0.1 * len(num_tags)
    iterations = 300
    burn_in = 200
    sample_every = 5 #based on the author thesis
    seed = 0 #time based seed
    lda_estimator = LDAEstimator(annotations_it, num_topics, alpha, beta, 
            gamma, iterations, burn_in, sample_every, seed)
    return lda_estimator

def create_bayes_estimator(annotations, lambda_, user_profile_fract_size=.4):
    '''
    Creates smooth estimator with the best Bayes parameter described in [1]_
    
    References
    ----------
    [1]_ Personalization of Tagging Systems, 
    Wang, Jun, Clements Maarten, Yang J., de Vries Arjen P., and 
    Reinders Marcel J. T. , 
    Information Processing and Management, Volume 46, Issue 1, p.58-70, (2010)
    '''
    smooth_estimator = SmoothEstimator('Bayes', lambda_, annotations,
                                       user_profile_fract_size)
    return smooth_estimator
