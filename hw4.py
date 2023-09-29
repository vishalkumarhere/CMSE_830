import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    return pd.read_csv(url, names=columns, sep=",\s*", engine="python")

data = load_data()

# Title and description
st.title("Census Income Prediction Web App")
st.write("Explore the dataset and predict income levels.")

# Sidebar for data exploration options
st.sidebar.header("Data Exploration Options")
selected_feature = st.sidebar.selectbox("Select Feature for Visualization", data.columns)
selected_chart = st.sidebar.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Scatter Plot"])
filtered_data = data[data["income"] == ">50K"] if st.sidebar.checkbox("Filter High Income") else data

# Data visualization
st.header("Data Exploration")
st.subheader("Summary Statistics")
st.write(data.describe())

st.subheader("Data Visualization")
if selected_chart == "Histogram":
    st.bar_chart(data[selected_feature].value_counts())
elif selected_chart == "Bar Chart":
    st.bar_chart(data[selected_feature].value_counts())
else:
    st.scatter_chart(data.sample(100), x="age", y="hours_per_week")

# Sidebar for income prediction
st.sidebar.header("Income Prediction")
model = st.sidebar.selectbox("Select Model", ["Random Forest"])
selected_features = st.sidebar.multiselect("Select Features for Prediction", data.columns[:-1])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Income prediction
st.header("Income Prediction")
if selected_features and st.sidebar.button("Train and Predict"):
    # Data preprocessing
    X = data[selected_features]
    y = data["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model training and prediction
    if model == "Random Forest":
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Display results
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

# GitHub link
st.sidebar.header("GitHub Repository")
st.sidebar.write("[GitHub Repository](https://github.com/yourusername/your-repo)")

# Footer
st.sidebar.header("About")
st.sidebar.write("This Streamlit app is for data exploration and income prediction using the Census Income dataset.")

# Data source
st.sidebar.header("Data Source")
st.sidebar.write("[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)")

