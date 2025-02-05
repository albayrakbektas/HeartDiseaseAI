import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, roc_auc_score, f1_score, recall_score, precision_score

data = pd.read_csv('data/heart_disease.csv')

print("Dataset size: ", data.shape)
print("Dataset head: ", data.head())

missing_values = data.isnull().sum()
print("Missing values: ", missing_values)

if missing_values.sum() > 0:
    print("Imputing missing values...")
    imputer = KNNImputer(n_neighbors=3)
    data.iloc[:, :] = imputer.fit_transform(data)

categorical_columns = ['sex', 'cp', 'restecg', 'slope', 'thal']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

print("Normalized numerical columns: ", data[numerical_columns].head())

sns.histplot(data['age'], kde=True)
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x='sex_1', hue='target', data=data)
plt.title("Heart Disease Distribution by Gender")
plt.xlabel("Gender (0 = Male, 1 = Female)")
plt.ylabel("Frequency")
plt.legend(["No Heart Disease", "Heart Disease"])
plt.show()

sns.boxplot(x="target", y="chol", data=data)
plt.title("Cholesterol Distribution by Target Variable")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Cholesterol (mg/dl)")
plt.show()

x = data.drop('target', axis=1)
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size: ", x_train.shape)
print("Testing set size: ", x_test.shape)

models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

param_grid = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30]
    },
    "Decision Tree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },
    "SVM": {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1, 10]
    }
}

optimized_models = {}

for model_name, model in models.items():
    if model_name in param_grid:
        print("\nPerforming hyperparameter optimization for {model_name}...}")
        grid = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(x_train, y_train)
        optimized_models[model_name] = grid.best_estimator_
        print(f"Best hyperparameters: {model_name}: {grid.best_params_}")
    else:
        model.fit(x_train, y_train)
        optimized_models[model_name] = model

results = {}
for model_name, model in optimized_models.items():
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    results[model_name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision,
        'ROC AUC': roc_auc
    }

results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)

plt.figure(figsize=(10, 8))
for model_name, model in optimized_models.items():
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

for model_name, model in optimized_models.items():
    cm = confusion_matrix(y_test, model.predict(x_test))
    ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()






















