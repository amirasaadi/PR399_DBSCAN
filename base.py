# #############################################################################
from matplotlib import pyplot as plt
# Generate sample data

from sklearn.datasets import make_blobs

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=.4,
                            random_state=0)
# #############################################################################
# Compute DBSCAN
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# #############################################################################
# Compute Performance Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


cm = confusion_matrix(labels_true, labels, normalize='true')
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(cmap='GnBu')
plt.savefig('cm_bolb.pdf', format='pdf')
print(classification_report(labels_true, labels, zero_division=0))

# #############################################################################
# Plotting Results

# Building the label to colour mapping
colours = {0: '#ff33a8', 1: '#a8ff33', 2: '#31a1f4', -1: '#808080'}

# Building the colour vector for each data point
cvec = [colours[label] for label in labels]

# For the construction of the legend of the plot
plt.figure()
scatter = plt.scatter(X[:, 0], X[:, 1], c=cvec, s=8)
# make figure square
plt.axis('square')
# make figures grid
plt.grid(color='black', linestyle=':', linewidth=0.8, alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bolbs data')
plt.savefig('bolb.pdf', format='pdf')
plt.show()