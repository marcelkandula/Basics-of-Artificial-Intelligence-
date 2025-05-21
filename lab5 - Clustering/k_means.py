import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    idx = np.random.choice(data.shape[0], size=k, replace=False)
    return data[idx]



def initialize_centroids_kmeans_pp(data, k):
    n_samples, n_features = data.shape
    centroids = np.empty((k, n_features))
    
    centroids[0] = data[np.random.randint(n_samples)]

    for i in range(1, k):
        min_distances = np.full(n_samples, np.inf)

        for j in range(i):
            centroid = centroids[j]
            diff = data - centroid
            dist_sq = np.sum(diff**2, axis=1)
            min_distances = np.minimum(min_distances, dist_sq)

        next_centroid_index = np.argmax(min_distances)
        centroids[i] = data[next_centroid_index]

    return centroids



import numpy as np

def assign_to_cluster(data, centroids):
    n_samples = data.shape[0]
    k = centroids.shape[0]
    distances = np.zeros((n_samples, k))

    for i in range(n_samples):
        for j in range(k):
            diff = data[i] - centroids[j]     
            squared_diff = np.dot(diff,diff)       
            sum_squared_diff = np.sum(squared_diff)  
            distance = np.sqrt(sum_squared_diff)    
            distances[i, j] = distance
    assignments = np.argmin(distances, axis=1)
    return assignments



def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    k = np.max(assignments) + 1
    n_features = data.shape[1]
    centroids = np.zeros((k, n_features))
    for i in range(k):
        points = data[assignments == i]
        if len(points) > 0:
            centroids[i] = np.mean(points, axis=0)
        else:
            centroids[i] = data[np.random.randint(data.shape[0])]
    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

