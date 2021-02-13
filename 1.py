# #############################################################################
from matplotlib import pyplot as plt
import numpy as np

# Generate sample data

n_samples = 750

X = np.random.rand(n_samples, 2), None
labels_true = np.zeros(750,dtype=int)
# #############################################################################
# Compute DBSCAN
from sklearn.cluster import DBSCAN

Epsilon = 0.3
MinPoint = 10
db = DBSCAN(eps=Epsilon, min_samples=MinPoint).fit(X[0])
labels = db.labels_

# #############################################################################
# Compute Performance Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


cm = confusion_matrix(labels_true, labels, normalize='true')
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot(cmap='GnBu')
plt.savefig('2-cm-NoStructure.pdf', format='pdf')
print(classification_report(labels_true, labels, zero_division=0))

# #############################################################################
# Plotting Results

# Building the label to colour mapping
colours = {0: '#ff33a8', 1: '#a8ff33', 2: '#31a1f4', -1: '#808080'}

# Building the colour vector for each data point
cvec = [colours[label] for label in labels]

# For the construction of the legend of the plot
plt.figure()
scatter = plt.scatter(X[0][:, 0], X[0][:, 1], c=cvec, s=8)
# make figure square
plt.axis('square')
# make figures grid
plt.grid(color='black', linestyle=':', linewidth=0.8, alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('No structure data')
plt.savefig('2-NoStructure.pdf', format='pdf')
plt.show()