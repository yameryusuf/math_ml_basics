import numpy as np
from matplotlib import pyplot as plt
def plot_param_posterior(lower_bound, upper_bound, blr, title, w0, w1, dist=0.1):
    fig = plt.figure()
    mesh_features, mesh_labels = np.mgrid[lower_bound:upper_bound:dist, lower_bound:upper_bound:dist]
    pos = np.dstack((mesh_features, mesh_labels))
    plt.contourf(mesh_features, mesh_labels, blr.P_ϴ_x_y.pdf(pos) )
    plt.scatter(w0, w1, color='red', label="True parameter values")
    plt.title(title)
    plt.xlabel("Intercept")
    plt.ylabel("Slope")
    plt.legend();
    return pos, blr.P_ϴ_x_y.pdf(pos)


def jitter(x):
    return x + np.random.uniform(low=-0.05, high=0.05, size=x.shape)

def plot_axis_pairs(X, axis_pairs, clusters, classes, dataset):
    n_rows = len(axis_pairs) // 2
    n_cols = 2
    plt.figure(figsize=(16, 10))
    for index, (x_axis, y_axis) in enumerate(axis_pairs):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.title('GMM Clusters')
        plt.xlabel(dataset.feature_names[x_axis])
        plt.ylabel(dataset.feature_names[y_axis])
        plt.scatter(
            jitter(X[:, x_axis]),
            jitter(X[:, y_axis]),
            c=clusters,
            cmap=plt.cm.get_cmap('brg'),
            marker='x')
    plt.tight_layout()


def plot_contour_pairs(X, y, colors):
    a = np.round( y, 2)
    b = np.random.randint(0,200,[1,1,a.shape[2],a.shape[2]])
    c = np.argmax(np.average(np.multiply(a[:, :, :, np.newaxis], b) / 255, axis=3), axis=2)
    d = np.multiply(a[:,:,:,np.newaxis] ,b)
    image = d[np.arange(d.shape[0])[:,None], np.arange(d.shape[1]), c].astype(np.uint8)
    print(type(image))
    plt.imshow(image, origin="lower")
    plt.title('GMM Clusters')
    for i in range(y.shape[2]):
        plt.figure(figsize=(16, 10))
        plt.title(f"GMM Clusters: Class ' {i}")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.contourf(X[:, :, 0], X[:, :, 1], y[:, :, i], 15, cmap=colors[i])
    plt.show()


def plot_axis(X, y, cdict):
    plt.figure(figsize=(16, 10))
    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        scatter = ax.scatter(
            jitter(X[:, 0][ix]),
            jitter(X[:, 1][ix]),
            c=cdict[g], label=g)
    ax.legend()
    plt.tight_layout()
    plt.show()