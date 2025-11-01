import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
target_names = iris.target_names
print("Features:", X.columns.tolist())
print("Target classes:", target_names)
print("Shape:", X.shape, y.shape)
print(y.value_counts())

# EDA Visualizations
sns.pairplot(pd.concat([X, y.rename('species')], axis=1), hue='species', corner=True)
plt.suptitle('Pairplot fitur Iris', y=1.02)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi antar fitur')
plt.show()

# Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# Logistic Regression Model
log_clf = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
log_clf.fit(X_train_scaled, y_train)
y_pred_log = log_clf.predict(X_test_scaled)

print("Logistic Regression - Classification Report")
print(classification_report(y_test, y_pred_log, target_names=target_names))

cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Decision Tree Model
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

print("Decision Tree - Classification Report")
print(classification_report(y_test, y_pred_dt, target_names=target_names))

plt.figure(figsize=(10,5))
plot_tree(dt_clf, feature_names=X.columns, class_names=target_names, filled=True, fontsize=8)
plt.title('Decision Tree Visualization')
plt.show()

cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Greens')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Metrics Comparison
def multi_reports(y_true, y_pred, average='macro'):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average)
    }

results = {
    'LogisticRegression': multi_reports(y_test, y_pred_log),
    'DecisionTree': multi_reports(y_test, y_pred_dt)
}
print(pd.DataFrame(results).T)

# Multiclass ROC Curve
y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_bin.shape[1]
y_score_log = log_clf.decision_function(X_test_scaled)
y_score_dt = dt_clf.predict_proba(X_test)

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_log[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {target_names[i]} (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Hyperparameter Tuning for Decision Tree
param_grid = {'max_depth': [None, 2, 3, 4, 5], 'min_samples_split':[2,4,6]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1_macro')
grid.fit(X_train, y_train)
print("Best params (DT):", grid.best_params_)
best_dt = grid.best_estimator_
y_pred_bestdt = best_dt.predict(X_test)
print(classification_report(y_test, y_pred_bestdt, target_names=target_names))

# Final comparison
final_results = {
    'LogisticRegression': multi_reports(y_test, y_pred_log),
    'DecisionTree': multi_reports(y_test, y_pred_dt),
    'DecisionTree_tuned': multi_reports(y_test, y_pred_bestdt)
}
print(pd.DataFrame(final_results).T)

import joblib
joblib.dump(log_clf, 'logistic_iris_model.joblib')
joblib.dump(best_dt, 'decisiontree_iris_model_tuned.joblib')
print("Models saved successfully!")
