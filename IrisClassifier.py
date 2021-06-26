import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from subprocess import check_output  (running this will list the files in the input directory)
# print(check_output(["ls", "D:/Python Programs/Neural Network"]).decode("utf8"))

iris = pd.read_csv("D:/Python Programs/Neural Network/iris.csv")
predict = pd.read_excel("D:/Python Programs/Neural Network/iris_predict.csv")
# print(iris.head()) (prints first 5 rows and 5 columns)
# print(iris.shape) (prints the shape of the .csv file)
# print(iris.dtypes) (prints the data types of the features and labels)

iris.iloc[:,0:3] = iris.iloc[:,0:3].astype(np.float32)
iris["species"] = iris["species"].map({"Iris-setosa":0, "Iris-virginica":1, "Iris-versicolor":2}) # Conversion

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(iris.iloc[:,0:4], iris["species"], test_size=0.33, random_state=42) #Split

columns = iris.columns[0:4]

import tensorflow as tf 
feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values, shape=[df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape=[labels.size,1])
    return feature_cols,label

classifier = tf.contrib.learn.DNNClassifier(
	feature_columns=feature_columns,
	hidden_units=[10],
	n_classes=3,
	optimizer=tf.train.GradientDescentOptimizer(0.001),
    activation_fn=tf.nn.relu
)

classifier.fit(input_fn=lambda:input_fn(xtrain,ytrain), steps=1000)
ev = classifier.evaluate(input_fn=lambda:input_fn(xtest,ytest), steps=1)
print(ev)

def input_predict(df):
    feature_cols = {k:tf.constant(df[k].values, shape=[df[k].size,1]) for k in columns}
    return feature_cols

pred = classifier.predict_classes(input_fn=lambda:input_predict(predict))
print(list(pred))
