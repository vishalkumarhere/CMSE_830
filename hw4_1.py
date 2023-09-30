import streamlit as st
import pandas as pd
import seaborn as sns

# Load the "adult" dataset from an online source
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
df_adult = pd.read_csv(url, header=None, names=column_names, sep=",\s*", engine="python")

# Streamlit app
st.title("Exploring the Adult Dataset")

# Display dataset summary
st.write("### Dataset Summary")
st.write(df_adult.head())

# Data exploration options
st.sidebar.title("Data Exploration Options")

# Show basic statistics
if st.sidebar.checkbox("Show Basic Statistics"):
    st.write("### Basic Statistics")
    st.write(df_adult.describe())

# Show a bar chart of education levels
if st.sidebar.checkbox("Education Level Counts"):
    st.write("### Education Level Counts")
    education_counts = df_adult["education"].value_counts()
    st.bar_chart(education_counts)

# Show a count of individuals by gender
if st.sidebar.checkbox("Gender Distribution"):
    st.write("### Gender Distribution")
    gender_counts = df_adult["sex"].value_counts()
    st.pie_chart(gender_counts)

# Show income distribution by education level
if st.sidebar.checkbox("Income by Education"):
    st.write("### Income Distribution by Education")
    income_by_education = df_adult.groupby("education")["income"].value_counts().unstack().fillna(0)
    st.bar_chart(income_by_education)

# Display a sample of the data
if st.sidebar.checkbox("Show Sample Data"):
    st.write("### Sample Data")
    sample_size = st.sidebar.slider("Number of Rows", 1, len(df_adult), 10)
    st.write(df_adult.sample(sample_size))

# You can add more data exploration options as needed

