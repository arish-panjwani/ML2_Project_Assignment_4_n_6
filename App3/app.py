import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# Title of the Streamlit app
st.title("Machine Learning: ANN with TanH Dropout")
st.sidebar.header("Options")

# Text loader for progress updates
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

    # Ensure y_pred_prob_binary is consistent
    y_pred_prob_binary = y_pred[:, 1] if len(y_pred.shape) > 1 else y_pred.ravel()

    # Check consistency
    st.write(f"y_test_binary shape: {y_test_binary.shape}")
    st.write(f"y_pred_prob_binary shape: {y_pred_prob_binary.shape}")

    # Compute ROC curve and AUC
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

    # Clear progress placeholder
    progress_placeholder.text("")
else:
    st.write("Please upload a CSV file to begin.")
