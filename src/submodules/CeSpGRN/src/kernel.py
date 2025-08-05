import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def calc_kernel_neigh(X, k = 5, bandwidth = 1, truncate = False, truncate_param = 30):

    # calculate pairwise distance
    D = squareform(pdist(X))
    for k in range(k, D.shape[0]):
        # calculate the knn graph from pairwise distance
        knn_index = np.argpartition(D, kth = k - 1, axis=1)[:, (k-1)]
        # find the value of the k-th distance
        kth_dist = np.take_along_axis(D, knn_index[:,None], axis = 1)
        # construct KNN graph
        knn = (D <= kth_dist)
        # make the graph symmetric
        knn = ((knn + knn.T) > 0).astype(np.int_)
        # knn = knn * knn.T
        # construct nextworkx graph from weighted knn graph, should make sure the G is connected
        G = nx.from_numpy_array(D * knn)
        # make sure that G is undirected
        assert ~nx.is_directed(G)
        # break while loop if connected
        if nx.is_connected(G):
            break
        else:
            k += 1
        print("number of nearest neighbor: " + str(k))

    # print("final number of nearest neighbor (make connected): " + str(k))
    # return a matrix of shortest path distances between nodes. Inf if no distances between nodes (should be no Inf, because the graph is connected)
    D = np.array(nx.floyd_warshall_numpy(G))
    assert np.max(D) < np.inf
    # scale the distance to between 0 and 1, similar to time-series kernel, values are still too large
    D = D/np.max(D)
    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    # if truncate the function
    if truncate == True:
        # trancate with the number of neighbors
        knn_index = np.argpartition(- K, kth = truncate_param - 1, axis=1)[:, (truncate_param-1)]
        kth_dist = np.take_along_axis(K, knn_index[:,None], axis = 1)
        mask = (K >= kth_dist).astype(np.int_)
        K_trun = K * mask
    else:
        K_trun = None
        
    # make the weight on each row sum up to 1
    return K/np.sum(K, axis = 1, keepdims = True), K_trun/np.sum(K_trun, axis = 1, keepdims = True)

def calc_kernel(X, k = 5, bandwidth = 1, truncate = False, truncate_param = 1):

    # calculate pairwise distance
    D = squareform(pdist(X))
    for k in range(k, D.shape[0]):
        # calculate the knn graph from pairwise distance
        knn_index = np.argpartition(D, kth = k - 1, axis=1)[:, (k-1)]
        # find the value of the k-th distance
        kth_dist = np.take_along_axis(D, knn_index[:,None], axis = 1)
        # construct KNN graph
        knn = (D <= kth_dist)
        # make the graph symmetric
        knn = ((knn + knn.T) > 0).astype(np.int)
        # knn = knn * knn.T
        # construct nextworkx graph from weighted knn graph, should make sure the G is connected
        G = nx.from_numpy_array(D * knn)
        # make sure that G is undirected
        assert ~nx.is_directed(G)
        # break while loop if connected
        if nx.is_connected(G):
            break
        else:
            k += 1
        print("number of nearest neighbor: " + str(k))

    print("final number of nearest neighbor (make connected): " + str(k))
    # return a matrix of shortest path distances between nodes. Inf if no distances between nodes (should be no Inf, because the graph is connected)
    D = nx.floyd_warshall_numpy(G)
    assert np.max(D) < np.inf
    # scale the distance to between 0 and 1, similar to time-series kernel, values are still too large
    D = D/np.max(D)
    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    # if truncate the function
    if truncate == True:
        print(mdis)
        print(np.sqrt(mdis))
        cutoff = np.sqrt(mdis) * truncate_param
        mask = (D < cutoff).astype(np.int)
        K_trun = K * mask
    else:
        K_trun = None
        
    # make the weight on each row sum up to 1
    return K/np.sum(K, axis = 1, keepdims = True), K_trun/np.sum(K_trun, axis = 1, keepdims = True)



def kernel_band(bandwidth, ntimes, truncate=False):

    # scale the t to be between 0 and 1
    t = (np.arange(ntimes)/ntimes).reshape(ntimes, 1)
    # calculate the pairwise-distance between time pointes
    D = np.square(pdist(t))
    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(D)
    # calculate the gaussian kernel function
    K = squareform(np.exp(-D/mdis))+np.identity(ntimes)

    # if truncate the function
    if truncate == True:
        cutoff = mdis * 1.5
        mask = (squareform(D) < cutoff).astype(np.int)
        K_trun = K * mask
    return K/np.sum(K, axis=1)[:, None], K_trun/np.sum(K_trun, axis = 1)[:, None]
