import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import SVC

# Load the csv file
data_file = pd.read_csv(r"C:\Users\casper\Desktop\iris.csv")
my_data= data_file[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
my_label = data_file["variety"]

# Split the dataset into train and test
my_data_train, my_data_test, my_label_train, my_label_test = train_test_split(my_data, my_label, test_size=0.3, random_state=50)
# The model
SVM_model = SVC()
# Fit the model
SVM_model.fit(my_data_train, my_label_train)

#Let's Make pickle file of our model
pickle.dump(SVM_model, open("model.pkl", "wb"))
print(SVM_Model.score(my_data,my_label))