import numpy
import scipy
from scipy.spatial.distance import pdist, squareform

def rbf(x,sigma):

	pairwise_dists = squareform(pdist(x, 'euclidean'));
	K = scipy.exp(pairwise_dists / sigma**2);
	return K;
