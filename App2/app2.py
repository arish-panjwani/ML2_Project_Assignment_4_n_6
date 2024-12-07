import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App
st.title("AdaBoost Classifier - Model Deployment")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Step 2: Preprocessing
    try:
        X = data.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]   # Target
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Encode the target variable
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        # Define the AdaBoost Classifier
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
        adaboost = AdaBoostClassifier(estimator=base_estimator, random_state=42)

        # Parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1],
        }

        # Hyperparameter tuning
        st.write("Running GridSearchCV...")
        grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        st.write("Best Parameters:", grid_search.best_params_)

        # Model training
        best_model.fit(X_train, y_train)

        # Predictions and Metrics
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("Classification Report:", pd.DataFrame(report).transpose())

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        # ROC-AUC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}")
        ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Curve")
        ax.legend()
        st.pyplot(fig)

        # Metrics Summary
        auc_score = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"AUC Score: {auc_score:.2f}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload a dataset to proceed.")
