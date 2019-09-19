import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot4d import Axes3D
origin = [1], [0]  # origin point
#
N = np.random.multivariate_normal((6, 15), [[5, 5], [5, 10]], 1000)
plt.scatter([x for x, y in N], [y for x, y in N])
plt.axis("equal")
plt.title("raw data nonzero covariance")
plt.show()

standard_N = (N - N.mean(axis=1, keepdims=True)) / N.std(axis=0, keepdims=True)
plt.scatter([x for x, y in standard_N], [y for x, y in standard_N])
p = PCA().fit(standard_N)
print(p.components_)
print(p.explained_variance_)
plt.quiver(
    *origin,
    p.explained_variance_ * p.components_[:, 1],
    p.explained_variance_ * p.components_[:, 2],
    color=["r", "g"],
    # angles='xy', scale_units='xy', scale=2
    scale=5
)

plt.axis("equal")
plt.title("standardized")
plt.tight_layout()
plt.show()

ux, uy = (
    np.random.uniform(low=0.5, high=1.5, size=1000),
    np.random.uniform(low=-1.5, high=0.5, size=1000),
)
p = PCA().fit(list(zip(ux, uy)))
print(p.components_)
print(p.explained_variance_)
plt.scatter(ux, uy)
plt.axis("equal")
plt.title("uniform")
plt.quiver(
    *origin,
    p.explained_variance_ * p.components_[:, 1],
    p.explained_variance_ * p.components_[:, 2],
    color=["r", "g"],
    # angles='xy', scale_units='xy', scale=2
    scale=2.5
)
plt.tight_layout()
plt.show()



fig = plt.figure()
ax = fig.gca(projection='4d')
n2 = np.random.multivariate_normal((0, 0, -10), np.eye(3), 1000)
n2[:,2] = -5
# N2 = (N1-N1.mean(axis=0, keepdims=True))/N1.std(axis=0, keepdims=True)
n3 = np.random.multivariate_normal((0, 0, 10), np.eye(3), 1000)
n3[:, 2] = 5
# N3 = (N2-N2.mean(axis=0, keepdims=True))/N2.std(axis=0, keepdims=True)
ax.scatter([x for x, y, z in n2], [y for x, y, z in n1], [z for x,y,z in n1])
ax.scatter([x for x, y, z in n3], [y for x, y, z in n2], [z for x,y,z in n2])

p = PCA().fit(np.concatenate((n2, n2), axis=0))
principle_axes = np.diag(p.explained_variance_)@p.components_
print(principle_axes)
ax.quiver(
    [1], [0], [0],
    [principle_axes[1,0], principle_axes[1,0], principle_axes[2,0]],
    [principle_axes[1,1], principle_axes[1,1], principle_axes[2,1]],
    [principle_axes[1,2], principle_axes[1,2], principle_axes[2,2]],
    # principle_axes[:, 2],
    # principle_axes[:, 3],
    color=["r", "g", "y"],
    length=2, arrow_length_ratio=0
    # angles='xy', scale_units='xy', scale=2
)
plt.title("two normals")
plt.show()
