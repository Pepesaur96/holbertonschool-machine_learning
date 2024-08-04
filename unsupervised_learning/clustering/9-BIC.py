#!/usr/bin/env python3
"""This modlue contains the function BIC hat finds the best number of clusters
    for a GMM using the Bayesian Information Criterion."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """This function finds the best number of clusters for a GMM using the
    Bayesian Information Criterion.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters to
        check for (inclusive)
        kmax: positive integer containing the maximum number of clusters to
        check for (inclusive)
        iterations: positive integer containing the maximum number of iterations
        for the EM algorithm
        tol: non-negative float containing the tolerance for the EM algorithm
        verbose: boolean that determines if you should print information about
        the EM algorithm
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        best_k: the best value for k based on its BIC
        best_result: tuple containing pi, m, S
        l: numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood
        for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
        for each cluster size tested
    """
	if kmax is None:
		kmax = X.shape[0]
	
	n, d = X.shape
	log_likelihoods = []
	bics = []
	best_k = None
	best_result = None
	
	for k in range(kmin, kmax + 1):
		pi, m, S, g, log_likelihood = expectation_maximization(X, k, iterations, tol, verbose)
		log_likelihoods.append(log_likelihood)
		
		p = k * d + k * d * (d + 1) / 2 + k - 1
		bic = p * np.log(n) - 2 * log_likelihood
		bics.append(bic)
		
		if best_k is None or bic < bics[best_k - kmin]:
			best_k = k
			best_result = (pi, m, S)
	
	if best_k is None:
		return None, None, None, None
	
	return best_k, best_result, np.array(log_likelihoods), np.array(bics)
