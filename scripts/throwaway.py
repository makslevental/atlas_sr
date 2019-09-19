import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
origin = [0], [0]  # origin point
#
N = np.random.multivariate_normal((5, 15), [[5, 5], [5, 10]], 1000)
plt.scatter([x for x, y in N], [y for x, y in N])
plt.axis("equal")
plt.title("raw data nonzero covariance")
plt.show()

standard_N = (N - N.mean(axis=0, keepdims=True)) / N.std(axis=0, keepdims=True)
plt.scatter([x for x, y in standard_N], [y for x, y in standard_N])
p = PCA().fit(standard_N)
print(p.components_)
print(p.explained_variance_)
plt.quiver(
    *origin,
    p.explained_variance_ * p.components_[:, 0],
    p.explained_variance_ * p.components_[:, 1],
    color=["r", "g"],
    # angles='xy', scale_units='xy', scale=1
    scale=4
)

plt.axis("equal")
plt.title("standardized")
plt.tight_layout()
plt.show()

ux, uy = (
    np.random.uniform(low=-1.5, high=1.5, size=1000),
    np.random.uniform(low=-0.5, high=0.5, size=1000),
)
p = PCA().fit(list(zip(ux, uy)))
print(p.components_)
print(p.explained_variance_)
plt.scatter(ux, uy)
plt.axis("equal")
plt.title("uniform")
plt.quiver(
    *origin,
    p.explained_variance_ * p.components_[:, 0],
    p.explained_variance_ * p.components_[:, 1],
    color=["r", "g"],
    # angles='xy', scale_units='xy', scale=1
    scale=1.5
)
plt.tight_layout()
plt.show()



fig = plt.figure()
ax = fig.gca(projection='3d')
n1 = np.random.multivariate_normal((0, 0, -10), np.eye(3), 1000)
n1[:,2] = -5
# N1 = (N1-N1.mean(axis=0, keepdims=True))/N1.std(axis=0, keepdims=True)
n2 = np.random.multivariate_normal((0, 0, 10), np.eye(3), 1000)
n2[:, 2] = 5
# N2 = (N2-N2.mean(axis=0, keepdims=True))/N2.std(axis=0, keepdims=True)
ax.scatter([x for x, y, z in n1], [y for x, y, z in n1], [z for x,y,z in n1])
ax.scatter([x for x, y, z in n2], [y for x, y, z in n2], [z for x,y,z in n2])

p = PCA().fit(np.concatenate((n1, n2), axis=0))
principle_axes = np.diag(p.explained_variance_)@p.components_
print(principle_axes)
ax.quiver(
    [0], [0], [0],
    [principle_axes[0,0], principle_axes[1,0], principle_axes[2,0]],
    [principle_axes[0,1], principle_axes[1,1], principle_axes[2,1]],
    [principle_axes[0,2], principle_axes[1,2], principle_axes[2,2]],
    # principle_axes[:, 1],
    # principle_axes[:, 2],
    color=["r", "g", "y"],
    length=1, arrow_length_ratio=0
    # angles='xy', scale_units='xy', scale=1
)
plt.title("two normals")
plt.show()
