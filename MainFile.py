# Data selection
import pandas as pd
df=pd.read_csv("Auto Sales data.csv")

# Pre processing 1

# Null value
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.isnull(),cmap="viridis",cbar=False)
plt.title("Missing Value")
plt.show()

# Pre processing 2

'''# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
columns_to_encode = [
    "ORDERDATE","STATUS", "PRODUCTLINE", "PRODUCTCODE", "CUSTOMERNAME", 
    "PHONE", "ADDRESSLINE1", "CITY", "POSTALCODE", "COUNTRY", 
    "CONTACTLASTNAME", "CONTACTFIRSTNAME","ORDERDATE"
]

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])'''
    
    
#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["ORDERDATE"]= le.fit_transform(df["ORDERDATE"])
df["STATUS"]= le.fit_transform(df["STATUS"])
df["PRODUCTLINE"]= le.fit_transform(df["PRODUCTLINE"])
df["PRODUCTCODE"]= le.fit_transform(df["PRODUCTCODE"])
df["CUSTOMERNAME"]= le.fit_transform(df["CUSTOMERNAME"])
df["PHONE"]= le.fit_transform(df["PHONE"])
df["ADDRESSLINE1"]= le.fit_transform(df["ADDRESSLINE1"])
df["CITY"]= le.fit_transform(df["CITY"])
df["POSTALCODE"]= le.fit_transform(df["POSTALCODE"])
df["COUNTRY"]= le.fit_transform(df["COUNTRY"])
df["CONTACTLASTNAME"]= le.fit_transform(df["CONTACTLASTNAME"])
df["CONTACTFIRSTNAME"]= le.fit_transform(df["CONTACTFIRSTNAME"])
df["DEALSIZE"]= le.fit_transform(df["DEALSIZE"])


# Data splitting
x=df.drop('DEALSIZE',axis=1)
y=df['DEALSIZE']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# Train the Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluate

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization

feature_idx = 1
plt.scatter(x_test.iloc[:, feature_idx], y_test, color='blue', label='Actual Data')
plt.scatter(x_test.iloc[:, feature_idx], y_pred, color='red', label='Predicted Data', alpha=0.7)
plt.title('Logistic Regression')
plt.xlabel(f'Feature: {x.columns[feature_idx]}')
plt.ylabel('Dependent Variable')
plt.legend()
plt.show()

'''num_features = 3  # Number of features to visualize
for feature_idx in range(num_features):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test.iloc[:, feature_idx], y_test, color='blue', label='Actual Data')
    plt.scatter(x_test.iloc[:, feature_idx], y_pred, color='red', alpha=0.7, label='Predicted Data')
    plt.title(f'Scatter Plot of Feature: {x.columns[feature_idx]} vs Target')
    plt.xlabel(f'Feature: {x.columns[feature_idx]}')
    plt.ylabel('Target (DEALSIZE)')
    plt.legend()
    plt.show()'''

import pickle

with open("Automobile_sales_model.pkl",'wb') as file:pickle.dump(model,file)

# Replace 'path_to_your_file.pkl' with the actual path to your PKL file
file_path = 'Automobile_sales_model.pkl'

# Open the file in binary mode and load the data
with open(file_path, 'rb') as file:data = pickle.load(file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)





