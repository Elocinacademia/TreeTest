# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv("../../../data/smarthome_plaintext.csv")
print("Read data ....")

#Drop nun values

df.dropna(inplace=True)
# One-hot encode the data using pandas get_dummies
df = pd.get_dummies(df)

# import pdb;pdb.set_trace()
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
train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size = 0.75, random_state = 42)

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



# if using decision tree

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.tree import _tree

df = DecisionTreeClassifier(max_depth=5, random_state=42)
model = df.fit(train_X, train_y)

# get the text representation
text_representation = tree.export_text(df, feature_names=feature_list)
print(text_representation)

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

rules = get_rules(df, feature_list, ["unaccept","accept"])
for r in rules:
    print(r)


# import pdb;pdb.set_trace()












# Instantiate model with 1000 decision trees
# clf = RandomForestClassifier(max_depth=10, random_state=42)
# clf = RandomForestClassifier(n_estimators=100)

# clf= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
#     max_depth=5, max_features='sqrt', max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_samples_leaf=1, min_samples_split=2,
#     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
#     oob_score=True, random_state=None, verbose=0,
#     warm_start=False)



# # Train the model on training data
# clf.fit(train_X, train_y)

# # checking the oob score
# print(clf.oob_score_)

# #Let’s do hyperparameter tuning for Random Forest using GridSearchCV and fit the data.
# clf = RandomForestClassifier(random_state=42, n_jobs=-1)
# params = {'max_depth': [2,3,5,10,20],
# 'min_samples_leaf': [5,10,20,50,100,200],
# 'n_estimators': [10,25,30,50,100,200]
# }


# from sklearn.model_selection import GridSearchCV
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=clf,param_grid=params,cv = 4,n_jobs=-1, verbose=1, scoring="accuracy")



# #%%time
# grid_search.fit(train_X, train_y)

# grid_search.best_score_

# rf_best = grid_search.best_estimator_
# rf_best

# from sklearn.tree import plot_tree
# plt.figure(figsize=(80,40))
# plot_tree(rf_best.estimators_[29], feature_names = feature_list,class_names=['Accept', "No Unaccept"],filled=True)

# import pdb;pdb.set_trace()

# Use the forest's predict method on the test data
predictions = df.predict(test_X)
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
# new_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', 
#     max_depth=None, max_features='sqrt', max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_samples_leaf=1, min_samples_split=2,
#     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#     oob_score=False, random_state=None, verbose=0,
#     warm_start=False)
# new_clf.fit(train_X, train_y)

# # prediction on test set
# y_pred = new_clf.predict(test_X)

# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Final Accuracy:",metrics.accuracy_score(test_y, y_pred))













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




