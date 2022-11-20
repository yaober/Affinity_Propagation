from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
 
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.4,
                            random_state=0)
 
 
def calc_similarity(X):
    m, n = np.shape(X)
    X_copy = np.copy(X)
    X = X.reshape((m, 1, n))
    X_copy = X_copy.reshape(1, m, n)
    sum = np.sum(np.square(X[..., :] - X_copy[..., :]), axis=-1)
    similarity = -1 * np.sqrt(sum)
    median_value = calc_median(similarity)
    for i in range(m):
        #set preference
        similarity[i, i] = median_value
    return similarity
 
 
def calc_median(X):
    data = []
    for i in range(len(X)):
        x = X[i]
        x = np.delete(x, i)
        data += list(x)
    n = len(data)
    return data[n // 2]
 
 
def affinityPropagation(similarity, lamda=0.):
    # matrix
    r = np.zeros_like(similarity, dtype=np.int32)
    a = np.zeros_like(similarity, dtype=np.int32)
 
    m, n = np.shape(similarity)
 
    last_change = np.zeros((m, ), dtype=np.int32)
    while True:
        # update r matrix
        for idx in range(m):
            a_s_idx = a[idx] + similarity[idx]
            for idy in range(n):
                a_s_idx_del = np.delete(a_s_idx, idy)
                max_value = np.max(a_s_idx_del)
                r_new = similarity[idx, idy] - max_value
                r[idx][idy] = lamda * r[idx][idy] + (1 - lamda) * r_new
        # update a matrix
        for idx in range(m):
            for idy in range(n):
                r_idy = r[:, idy]
                r_idy = np.delete(r_idy, idx)
                a_new = np.sum(np.maximum(0, r_idy))
                if idx != idy:
                    a_new = min(0, r[idy, idy] + a_new)
                a[idx][idy] = lamda * a[idx][idy] + (1 - lamda) * a_new
        r_a = r + a
        # no change->stop
        argmax = np.argmax(r_a, axis=1)
        current_change = argmax
        if (last_change == current_change).all():
            break
        last_change = current_change
    print('r', r)
    print('a', a)
    return r + a
 
 
def computeCluster(fitable, data):
    clusters = {}
    num = len(fitable)
    for idx in range(num):
        fit = fitable[idx]
        argmax = np.argmax(fit, axis=-1)
        if argmax not in clusters:
            clusters[argmax] = []
        clusters[argmax].append(tuple(data[idx]))
    return clusters
 
 
def plotClusters(clusters, title):
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    col = []
    r = lambda: np.random.randint(0, 255)
    for index in range(len(clusters)):
        col.append(('#%02X%02X%02X' % (r(), r(), r())))
    color = 0
    for key in clusters:
        cluster = clusters[key]
        for idx in range(len(cluster)):
            cluster_idx = cluster[idx]
            axes.scatter(cluster_idx[0], cluster_idx[1], s=20, c=col[color])
        color += 1
    plt.title(title)
    plt.show()
 
 
similarity = calc_similarity(X)
fitable = affinityPropagation(similarity, lamda=0.25)
clusters = computeCluster(fitable, X)
print(len(clusters))
plotClusters(clusters, "clusters by affinity propagation")