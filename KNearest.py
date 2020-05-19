import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[2,4], [4,2], [4,4], [4,6], [6,2], [6,4]])
y = np.array([0, 0, 1, 0, 1, 0])

plt.scatter(X[:,0], X[:,1])
plt.show()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

array = np.array([[6, 6]])
print(f"Class of point (6,6) is : {model.predict(array)}")

