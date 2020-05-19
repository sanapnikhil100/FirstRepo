import matplotlib.pyplot as m
import pandas as p
import numpy as np

def main():

	dataset = p.read_csv("tree.csv")
	x = dataset.iloc[:, 1:-1] #x is test data to predict buys i.e. 5th col
	y = dataset.iloc[:, 5] #select all rows and 5th col i.e. y contains only buys
	#i.e. selecting only buy col

	x = x.values
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	from sklearn.compose import ColumnTransformer
	lab_enc = LabelEncoder()
	x[:, 1] = lab_enc.fit_transform(x[:, 1])
	lab_enc = LabelEncoder()
	x[:, 2] = lab_enc.fit_transform(x[:, 2])
	lab_enc = LabelEncoder()
	x[:, 3] = lab_enc.fit_transform(x[:, 3])
	col_trans = ColumnTransformer([("Age", OneHotEncoder(), [0])], remainder='passthrough')
	x = col_trans.fit_transform(x)
	x = p.DataFrame(x)
	print (x)
	lab_enc = LabelEncoder()
	y = lab_enc.fit_transform(y)
	y = p.DataFrame(y)
	print (y)

	from sklearn.tree import DecisionTreeClassifier
	model = DecisionTreeClassifier()
	model.fit(x, y)

	print (model)

	input = np.array([0, 1, 0, 1, 0, 0])
	output = model.predict([input])
	print ("output:", output)


if __name__ == "__main__":
	main()
