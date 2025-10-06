import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Title for the web page
st.title("Crop Recommendation using Machine Learning")

# Sidebar menu
select_value = st.sidebar.radio("Menu", options=["Home", "Data Upload", "Data Preprocess","Visualizations","Model Training", "Testing", "Evaluation"])

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

if select_value == "Data Upload":
    st.subheader("Uploaded Data Preview")
    st.write(data.head())  # Displays the first 5 rows in Streamlit



# Creating a Label Encoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

if select_value == "Data Preprocess":
    st.subheader("Data Preprocessing & Analysis")
    st.write("### Data Info:")
    st.text(data.info())

    st.write("### Data Description:")
    st.write(data.describe())

    st.write("### Correlation Matrix:")
    st.write(data.corr())

    st.write(f"### Data Shape: {data.shape}")
if select_value == "Visualizations":
   
    # Visualizing Crop Distribution
    st.subheader("Crop Distribution")
    vc = data['label'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(vc.index, vc.values)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    # Scatter Plot for pH vs Label
    st.subheader("pH vs Crop Type")
    fig, ax = plt.subplots()
    ax.scatter(data['ph'], data['label'])
    plt.xlabel("pH")
    plt.ylabel("Crop Label")
    st.pyplot(fig)

    # Histogram for Temperature Distribution
    st.subheader("Temperature Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['temperature'], bins=20, color="blue", alpha=0.7)
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")
    st.pyplot(fig)

# Splitting Data
X = data.iloc[:, :-1].values  # Input features
Y = data.iloc[:, -1].values   # Output (Crop Type)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.35, random_state=42)

# Model Training
if select_value == "Model Training":
    st.subheader("Training SVM Model")
    model = SVC()
    model.fit(Xtrain, Ytrain)
    joblib.dump(model, "svm_crop_model.pkl")  # Save the model
    st.success("Model Training Completed!")

# Model Evaluation
if select_value == "Evaluation":
    st.subheader("Model Evaluation")
    model = joblib.load("svm_crop_model.pkl")  # Load model
    pred = model.predict(Xtest)

    acc = accuracy_score(Ytest, pred)
    prec = precision_score(Ytest, pred, average='micro')
    re = recall_score(Ytest, pred, average='micro')

    st.write(f"**Accuracy:** {acc*100:.2f}%")
    st.write(f"**Precision:** {prec*100:.2f}%")
    st.write(f"**Recall:** {re*100:.2f}%")

    # Bar chart for model performance
    st.subheader("Model Performance Metrics")
    metrics = ["Accuracy", "Precision", "Recall"]
    scores = [acc * 100, prec * 100, re * 100]

    fig, ax = plt.subplots()
    ax.bar(metrics, scores, color=["green", "blue", "red"])
    plt.ylim(0, 100)
    st.pyplot(fig)

# New Prediction
if select_value == "Testing":
    st.subheader("Test with New Input")
    model = joblib.load("svm_crop_model.pkl")  # Load model
    N = st.number_input("Nitrogen", value=85)
    P = st.number_input("Phosphorus", value=58)
    K = st.number_input("Potassium", value=41)
    temperature = st.number_input("Temperature", value=21.77)
    humidity = st.number_input("Humidity", value=80.32)
    ph = st.number_input("pH Level", value=7.03)
    rainfall = st.number_input("Rainfall", value=226.66)

    if st.button("Predict Crop"):
        xnew = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        output = model.predict(xnew)
        predicted_crop = le.inverse_transform(output)[0]
        st.success(f"Recommended Crop: {predicted_crop}")

