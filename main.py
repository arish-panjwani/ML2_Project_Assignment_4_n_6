import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score, auc
)
from sklearn.model_selection import GridSearchCV
from joblib import load
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import time

# Preprocessing function for App1
def preprocess_dataframe(df):
    df_clean = df.copy()

    # Columns to convert TRUE/FALSE to 1/0
    columns_to_convert = [
        'BusinessAcceptsCreditCards', 'RestaurantsPriceRange2', 'RestaurantsTakeOut',
        'RestaurantsDelivery', 'Caters', 'WiFi', 'WheelchairAccessible',
        'OutdoorSeating', 'HasTV', 'RestaurantsReservations', 'GoodForKids',
        'RestaurantsGoodForGroups'
    ]

    for col in columns_to_convert:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace({
                "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0,
                True: 1, False: 0,
                "": 0, "nan": 0, np.nan: 0
            })
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    # Handle WiFi column
    if 'WiFi' in df_clean.columns:
        df_clean['WiFi'] = df_clean['WiFi'].astype(str)
        wifi_mapping = {'0': 0, 'no': 0, 'free': 1, 'paid': 2}
        df_clean['WiFi'] = df_clean['WiFi'].replace(wifi_mapping)
        df_clean['WiFi'] = pd.to_numeric(df_clean['WiFi'], errors='coerce').fillna(0).astype(int)

    # Drop irrelevant columns
    columns_to_drop = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'categories']
    df_clean = df_clean.drop(columns=[c for c in columns_to_drop if c in df_clean.columns], errors='ignore')

    # Select numeric columns and fill missing
    df_numeric = df_clean.select_dtypes(include=[np.number]).fillna(0)

    return df_numeric

# Align features dynamically
def align_features(df, model_features):
    """
    Ensures that the DataFrame has the exact features required by the model.
    - Adds missing features with default values (0).
    - Reorders features to match the model's training features.
    """
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value
    return df[model_features]  # Return aligned DataFrame

# Function for App1 (Anomaly Detection)
def app1():
    st.title("Anomaly Detection with Multiple Models")

    # Define features required for OneClassSVM
    one_class_svm_features = [
        'latitude', 'longitude', 'review_count', 'BusinessParking_garage',
        'BusinessParking_lot', 'BusinessParking_valet', 'BusinessAcceptsCreditCards',
        'BikeParking', 'RestaurantsPriceRange2', 'RestaurantsTakeOut', 'WiFi',
        'RestaurantsDelivery', 'Caters'
    ]

    # Load models
    models = {}
    error_messages = []
    try:
        models['DBSCAN'] = load('dbscan_model.pkl')
    except FileNotFoundError:
        error_messages.append("DBSCAN model is not loaded.")

    try:
        models['IsolationForest'] = load('isolation_forest_model.pkl')
    except FileNotFoundError:
        error_messages.append("Isolation Forest model is not loaded.")

    try:
        models['OneClassSVM'] = load('one_class_svm_model.pkl')
    except FileNotFoundError:
        error_messages.append("OneClassSVM model is not loaded.")

    try:
        models['LOF'] = load('lof_model.pkl')
        if models['LOF'] is None:
            raise ValueError("LOF model is not properly loaded.")
    except (FileNotFoundError, ValueError):
        error_messages.append("LOF model is not properly loaded.")

    # Display success or error messages related to models
    if error_messages:
        st.warning("Some models were not loaded successfully:")
        for msg in error_messages:
            st.write(f"- {msg}")
    else:
        st.success("All models loaded successfully!")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

    st.write("**Dataset Preview:**")
    st.dataframe(data.head())

    # Preprocess data
    st.text("Preprocessing data...")
    X = preprocess_dataframe(data)

    # Generate predictions
    results_df = data.copy()
    model_comparison = {}

    # Initialize progress message
    progress_message = st.empty()

    if 'DBSCAN' in models:
        try:
            progress_message.text("Running DBSCAN...")
            dbscan_labels = models['DBSCAN'].fit_predict(X)
            results_df['DBSCAN_label'] = dbscan_labels
            model_comparison['DBSCAN'] = sum(dbscan_labels == -1)  # Count of anomalies (-1)
        except Exception as e:
            st.error(f"DBSCAN failed: {e}")

    if 'IsolationForest' in models:
        try:
            progress_message.text("Running Isolation Forest...")
            X_isolation_forest = align_features(X, models['IsolationForest'].feature_names_in_)
            if_labels = models['IsolationForest'].predict(X_isolation_forest)
            results_df['IsolationForest_label'] = if_labels
            model_comparison['IsolationForest'] = sum(if_labels == -1)  # Count of anomalies (-1)
        except Exception as e:
            st.error(f"Isolation Forest failed: {e}")

    if 'OneClassSVM' in models:
        try:
            progress_message.text("Running OneClassSVM...")
            # Align features for OneClassSVM
            X_one_class_svm = align_features(X, one_class_svm_features)
            ocsvm_labels = models['OneClassSVM'].predict(X_one_class_svm)
            results_df['OneClassSVM_label'] = ocsvm_labels
            model_comparison['OneClassSVM'] = sum(ocsvm_labels == -1)  # Count of anomalies (-1)
        except Exception as e:
            st.error(f"OneClassSVM failed: {e}")

    if 'LOF' in models:
        try:
            progress_message.text("Running LOF...")
            lof_labels = models['LOF'].fit_predict(X)
            results_df['LOF_label'] = lof_labels
            model_comparison['LOF'] = sum(lof_labels == -1)  # Count of anomalies (-1)
        except Exception as e:
            st.error(f"LOF failed: {e}")

    progress_message.text("Processing completed.")

    # Display results
    st.write("**Model Results:**")
    st.dataframe(results_df.head())

    # Comparison Chart
    st.write("**Anomaly Count Comparison:**")
    if model_comparison:
        comparison_df = pd.DataFrame.from_dict(model_comparison, orient='index', columns=['Anomaly Count'])
        st.bar_chart(comparison_df)
    else:
        st.write("No model outputs available for comparison.")

    # Download button for results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='anomaly_detection_results.csv',
        mime='text/csv',
    )


# Function for App 2 (AdaBoost Classifier)
def app2():
    st.title("AdaBoost Classifier - Model Deployment")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())

        try:
            # Preprocessing
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

# Function for App 3 (Updated with provided code)
def app3():
    st.title("Machine Learning: ANN with TanH Dropout")
    st.sidebar.header("Options")

    progress_placeholder = st.empty()

    # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Step 2: Load the dataset
        progress_placeholder.text("Loading dataset...")
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())

        # Fill missing values
        progress_placeholder.text("Handling missing values...")
        df.fillna(df.mean(), inplace=True)

        # One-hot encoding categorical features
        progress_placeholder.text("Encoding categorical features...")
        cities = [col for col in df.columns if col.startswith('city_')]
        states = [col for col in df.columns if col.startswith('state_')]
        df = pd.get_dummies(df, columns=cities + states, drop_first=True)

        # Categorize 'stars'
        progress_placeholder.text("Categorizing 'stars' column...")
        bins = [0, 2, 4, 5]
        labels = [0, 1, 2]
        df['stars'] = pd.cut(df['stars'], bins=bins, labels=labels)

        # Split dataset
        progress_placeholder.text("Splitting dataset into train and test sets...")
        X = df.drop(columns=['stars'])
        y = df['stars'].astype('int64')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

        # Build the model
        progress_placeholder.text("Building ANN model with TanH activation and Dropout...")
        def build_model(input_dim, output_dim):
            model = models.Sequential([
                layers.Dense(64, activation='tanh', input_dim=input_dim),
                layers.Dropout(0.3),
                layers.Dense(128, activation='tanh', kernel_regularizer=regularizers.l2(0.001)),
                layers.Dropout(0.3),
                layers.Dense(output_dim, activation='softmax' if output_dim > 1 else 'sigmoid')
            ])
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy',
                          metrics=['accuracy'])
            return model

        model = build_model(X_train.shape[1], len(np.unique(y)))

        # Train the model
        progress_placeholder.text("Training the model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
        end_time = time.time()
        train_time = end_time - start_time
        st.write(f"Training Time: {train_time:.2f} seconds")

        # Save the model
        progress_placeholder.text("Saving the trained model...")
        model.save('trained_model.h5')

        # Evaluation Metrics
        progress_placeholder.text("Evaluating the model...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Accuracy: {test_acc:.2f}")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        st.write(f"F1 Score (Weighted): {f1:.2f}")

        # ROC Curve and AUC
        progress_placeholder.text("Generating ROC curve...")
        y_test_binary = (y_test == 1).astype(int)
        y_pred_prob_binary = y_pred[:, 1] if len(y_pred.shape) > 1 else y_pred.ravel()

        try:
            fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob_binary)
            roc_auc = auc(fpr, tpr)
            st.write(f"AUC: {roc_auc:.2f}")

            # Plot ROC Curve
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Model')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)
        except ValueError as e:
            st.error(f"Error computing ROC curve: {e}")

        progress_placeholder.text("")
    else:
        st.write("Please upload a CSV file to begin.")

# Main app
def main():
    st.sidebar.title("Application Selector")
    app_choice = st.sidebar.selectbox("Select an App", ["Unsupervised Models (IF/LOF/DBSc/OcSVM)", "Adaboost", "Ann with TanH Dropout"])

    if app_choice == "Unsupervised Models (IF/LOF/DBSc/OcSVM)":
        app1()
    elif app_choice == "Adaboost":
        app2()
    elif app_choice == "Ann with TanH Dropout":
        app3()

if __name__ == "__main__":
    main()
