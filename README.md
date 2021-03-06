# Iris Classification
Classification of 3 species of flowers (versicolor, virginica, setosa) belonging to the Iris family, using a Fully Connected Neural Network for Data Processing (Tensorflow 1.12.0 and Python 3.6.6)

- Editor used: Sublime Text 3
- Shell used to run the code: Git Bash
- Libraries used: Pandas, Numpy & Tensorflow
- iris.csv is the main dataset, which is used for the training and testing stages of the model
- iris_predict.csv is the prediction dataset, which is used as input for the prediction stage of the model after the training and testing stages are completed

Insight on iris.csv
- There are a total of 5 columns of data
- The first 4 columns serve as the features for the model
- The last (5th) column serves as the result, which the model predicts and trains itself on during the prediction and training+testing stages respectively. The iris_predict.csv file does not contain the 5th/result column since the model is supposed to predict that result and generate the same as it's output.
- Column 1: Represents the sepal length of an individual flower
- Coumn 2: Represents the sepal width of an individual flower
- Column 3: Represents the petal length of an individual flower
- Column 4: Represents the petal width of an individual flower
- Column 5: Represents the species of an individual flower
