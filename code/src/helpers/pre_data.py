import pandas as pd
import numpy as np

df = pd.read_csv("../../../data/data.csv")
print("Read data ....")

#Drop nun values

df.dropna(inplace=True)
# One-hot encode the data using pandas get_dummies
df = pd.get_dummies(df)

# Labels are the values we want to predict
labels = np.array(df['Class'])

# Remove the labels from the df
# axis 1 refers to the columns
df = df.drop('Class', axis = 1)

# Saving feature names for later use
feature_list = list(df.columns)

# Convert to numpy array
df = np.array(df)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size = 0.2, random_state = 42)

print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_y.shape)
print('Testing Features Shape:', test_X.shape)
print('Testing Labels Shape:', test_y.shape)


# # The baseline predictions are the historical averages
# baseline_preds = test_X[:, feature_list.index('average')]

# # Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - test_y)
# print('Average baseline error: ', round(np.mean(baseline_errors), 2))


# Import the model we are using
print("Load model ....")
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees
# clf = RandomForestClassifier(max_depth=10, random_state=42)
# clf = RandomForestClassifier(n_estimators=100)
clf= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
    max_depth=None, max_features='sqrt', max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
    oob_score=False, random_state=None, verbose=0,
    warm_start=False)
# Train the model on training data
clf.fit(train_X, train_y)

# Use the forest's predict method on the test data
predictions = clf.predict(test_X)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Original Model Accuracy:",metrics.accuracy_score(test_y, predictions))


''' ===========
Result printing:
==========='''
# clf.estimators_[100]
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig = plt.figure(figsize=(15, 10))
plot_tree(clf.estimators_[0], 
          feature_names=feature_list,
          class_names='Class', 
          filled=True, impurity=True, 
          rounded=True)

fig.savefig('figure_name.png')


import pdb; pdb.set_trace()     









import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=feature_list).sort_values(ascending=False)
feature_imp_top_15 = [x for x in feature_imp.index][:5]

# import matplotlib.pyplot as plt

# import seaborn as sns
# # Creating a bar plot
# sns.barplot(x=feature_imp, y=feature_imp.index)
# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()






#Generate model with selected features
# Import train_test_split function
# Split dataset into features and labels
X = df
y = labels
data = pd.read_csv("../../../data/data.csv")
print("Reload data ....")

#Drop nun values
data.dropna(inplace=True)
# One-hot encode the data using pandas get_dummies
data = pd.get_dummies(data)


# Remove the labels from the data df
# axis 1 refers to the columns
data = data.drop('Class', axis = 1)

X = data[feature_imp_top_15]

                 
# Split dataset into training set and test set
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5) # 70% training and 30% test



# new_clf = RandomForestClassifier(n_estimators=100)

# new_clf = RandomForestClassifier(n_estimators=100, 
#     max_depth=3,
#     max_features='sqrt', 
#     min_samples_leaf=4,
#     bootstrap=True, 
#     n_jobs=-1, 
#     random_state=0)
new_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
    max_depth=None, max_features='sqrt', max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
    oob_score=False, random_state=None, verbose=0,
    warm_start=False)
new_clf.fit(train_X, train_y)

# prediction on test set
y_pred = new_clf.predict(test_X)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Final Accuracy:",metrics.accuracy_score(test_y, y_pred))













#! When you predict numerical values and have some baseline to prepare:
# # Calculate the absolute errors
# errors = abs(predictions - test_y)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2))





# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_y)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')




