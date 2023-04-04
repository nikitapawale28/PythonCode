from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

filename = 'diabetes.csv'
df = read_csv(filename)

df.head()
print(df.shape)
array = df.values
X1 = array[:, 0:8]
y1 = array[:, 8]
kfold1 = KFold(n_splits=5, random_state=2, shuffle=True)
model = KNeighborsClassifier(n_neighbors=3)
results = cross_val_score(model, X1, y1, cv=kfold1)
print("Without any changes accuracy of knn.py", results.mean())
######################data cleaning part####################################################################################
# data has some missing values so filling them with either mean or median values
diabetes_data_Modified = df.copy(deep=True)
diabetes_data_Modified[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_Modified[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

diabetes_data_Modified['Glucose'].fillna(diabetes_data_Modified['Glucose'].mean(), inplace=True)
diabetes_data_Modified['BloodPressure'].fillna(diabetes_data_Modified['BloodPressure'].mean(), inplace=True)
diabetes_data_Modified['SkinThickness'].fillna(diabetes_data_Modified['SkinThickness'].median(), inplace=True)
diabetes_data_Modified['Insulin'].fillna(diabetes_data_Modified['Insulin'].median(), inplace=True)
diabetes_data_Modified['BMI'].fillna(diabetes_data_Modified['BMI'].median(), inplace=True)

diabetes_data_Modified.head()
print(diabetes_data_Modified.shape)
array = diabetes_data_Modified.values
X1 = array[:, 0:8]
y1 = array[:, 8]
kfold1 = KFold(n_splits=5, random_state=2, shuffle=True)
model = KNeighborsClassifier(n_neighbors=3)
results = cross_val_score(model, X1, y1, cv=kfold1)
print("Accuracy after data cleaning ", results.mean())
################################Feature selection by data correlation#########################################
import seaborn as sns

data_corr = diabetes_data_Modified.corr()
# sns.heatmap(data_corr, vmax=1, square=True)
sns.heatmap(data_corr, annot=True, cmap='RdYlGn')
plt.title("Correlation of Data")
plt.show()

# highly correlated with output are glucose,BMI and age.
# less correlated with output are  blood pressure, insulin, DiabetesPedigreeFunction, pregnancies, skin thickness.
# However, Glucose is higly correlated to insulin and skin thickness is highly corrlated to BMI
# so we can skip the features like pregnancies,blood pressure and DiabetesPedigreeFunction

diabetes_data_Modified = diabetes_data_Modified.drop(
    columns=['BloodPressure', 'DiabetesPedigreeFunction', 'Pregnancies'])
print(diabetes_data_Modified.shape)
##############################Check accuracy after feature selection####################################################################################
array = diabetes_data_Modified.values
X = array[:, 0:5]
y = array[:, 5]
k_range = range(1, 11)
c_range = range(2, 12)
k_scores = []
listX = []
listY = []

for k in k_range:
    knn = KNeighborsClassifier(k)
    for c in c_range:
        kfold = KFold(n_splits=c, random_state=2, shuffle=True)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
        listX.append(k)
        listY.append(scores.mean())
plt.scatter(listX, listY)
plt.xlabel('K-value')
plt.ylabel('Cross-validated accuracy')
plt.show()
print("Maximum Accuracy for Cross-validatation with feature selection is", max(listY), " for K value ", listX[listY.index(max(listY))])
listX = []
listY = []

###############################With MinMax Scalar####################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

df_model = diabetes_data_Modified.copy()

features = list(df_model.columns)
features = [features[:-1]]
scaler = MinMaxScaler()
for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

knn = KNeighborsClassifier()

# Create X and y variable
X = df_model.drop(columns=['Outcome'])
y = df_model['Outcome']
for k in k_range:
    knn = KNeighborsClassifier(k)
    for c in c_range:
        kfold = KFold(n_splits=c, random_state=2, shuffle=True)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
        listX.append(k)
        listY.append(scores.mean())
plt.scatter(listX, listY)
plt.xlabel('K-value')
plt.ylabel('Cross-validated accuracy for MinMaxScalar')
plt.show()
print("Maximum Accuracy for MinMaxScalar Scalar is", max(listY), " for K value ", listX[listY.index(max(listY))])
listX = []
listY = []

################robust scalar###############################
df_model = diabetes_data_Modified.copy()

features = list(df_model.columns)
# print(features)
features = [features[:-1]]
scaler = RobustScaler()
for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

# Create X and y variable
X = df_model.drop(columns=['Outcome'])
y = df_model['Outcome']
for k in k_range:
    knn = KNeighborsClassifier(k)
    for c in c_range:
        kfold = KFold(n_splits=c, random_state=2, shuffle=True)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
        # print("K", k, "scores", scores)
        # k_scores.append(scores.mean())

        listX.append(k)
        listY.append(scores.mean())
plt.scatter(listX, listY)

plt.xlabel('K-value')
plt.ylabel('Cross-validated accuracy for RobusScalar')
plt.show()
print("Maximum Accuracy for Robust Scalar is", max(listY), " for K value ", listX[listY.index(max(listY))])
listX = []
listY = []

#####################standard Scalar#############################
warnings.filterwarnings('ignore')

df_model = diabetes_data_Modified.copy()

features = list(df_model.columns)
# print(features)
features = [features[:-1]]
scaler = StandardScaler()

for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])
# Create X and y variable
X = df_model.drop(columns=['Outcome'])
y = df_model['Outcome']
for k in k_range:
    knn = KNeighborsClassifier(k)
    for c in c_range:
        kfold = KFold(n_splits=c, random_state=2, shuffle=True)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
        # print("K", k, "scores", scores)
        # k_scores.append(scores.mean())

        listX.append(k)
        listY.append(scores.mean())
plt.scatter(listX, listY)

plt.xlabel('K-value')
plt.ylabel('Cross-validated accuracy for Standard Scalar')
plt.show()
print("Maximum Accuracy for standard Scalar is", max(listY), " for K value ", listX[listY.index(max(listY))])
listX = []
listY = []
#######################quantile Transform########################################

df_model = diabetes_data_Modified.copy()

features = list(df_model.columns)
# print(features)
features = [features[:-1]]
scaler = QuantileTransformer(random_state=0)
for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

# Create X and y variable
X = df_model.drop(columns=['Outcome'])
y = df_model['Outcome']
for k in k_range:
    knn = KNeighborsClassifier(k)
    for c in c_range:
        kfold = KFold(n_splits=c, random_state=2, shuffle=True)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')

        listX.append(k)
        listY.append(scores.mean())
plt.scatter(listX, listY)

plt.xlabel('K-value')
plt.ylabel('Cross-validated accuracy for QuantileTransformer')
plt.show()
print("Maximum Accuracy for QuantileTransformer is", max(listY), " for K value ", listX[listY.index(max(listY))])
listX = []
listY = []
############################Power transformer Scalar##############################
df_model = diabetes_data_Modified.copy()

features = list(df_model.columns)
# print(features)
features = [features[:-1]]
scaler = PowerTransformer()
for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

# Create X and y variable
X = df_model.drop(columns=['Outcome'])
y = df_model['Outcome']
for k in k_range:
    knn = KNeighborsClassifier(k)
    for c in c_range:
        kfold = KFold(n_splits=c, random_state=2, shuffle=True)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')

        listX.append(k)
        listY.append(scores.mean())
plt.scatter(listX, listY)

plt.xlabel('K-value')
plt.ylabel('Cross-validated accuracy for PowerTransformer')
plt.show()
print("Maximum Accuracy for PowerTransformer is", max(listY), " for K value ", listX[listY.index(max(listY))])