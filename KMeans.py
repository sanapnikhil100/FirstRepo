import matplotlib.pyplot as m
import pandas as p
import numpy as np

def main():
	x=[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3]
	y=[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]
        #plotting values of x,y on graph

	m.scatter(x, y, color = 'r')

	#fake centroids
	cx = np.array([0.1,0.3])
	cy = np.array([0.6,0.2])

	m.scatter(cx[0], cy[0], color = 'g')
	m.scatter(cx[1], cy[1], color = 'g')
	m.show()

	X = np.array(list(zip(x,y)))
	#print (X)
	

	startpts=np.array([[0.1, 0.6], [0.3, 0.2]], np.float64)
	from sklearn.cluster import KMeans
	model = KMeans(n_clusters = 2, init=startpts, n_init=1)
	model.fit(X)
	centroids = model.cluster_centers_

	m.scatter(x, y, color = 'r')
	m.scatter(centroids[:,0], centroids[:,1], c='g')
	m.show()

	array = np.array([[0.25, 0.5]])
	get_cluster = model.predict(array)
	print(f"Point P6 belongs to cluster : {get_cluster}")

	labels = model.labels_
	count = np.count_nonzero(labels == 1)
	print(f"Population around cluster around m2 is {count}")
	print (centroids[0])
	print (centroids[1])


if __name__ == "__main__":
        main()



