import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the "adult" dataset from Seaborn
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    return pd.read_csv(url, names=columns, sep=",\s*", engine="python")

data = load_data()

# Set the title and description for the web app
st.title("Exploring the Adult Dataset")
st.write("This web app allows you to explore the 'adult' dataset from Seaborn.")

# Sidebar for filtering options
st.sidebar.header("Filter Data")
age_range = st.sidebar.slider("Select Age Range", min_value=0, max_value=100, value=(0, 100))
education_levels = st.sidebar.multiselect("Select Education Levels", df_adult["education"].unique())
income_level = st.sidebar.selectbox("Select Income Level", df_adult["income"].unique())

# Apply filters to the dataset
filtered_df = df_adult[
    (df_adult["age"] >= age_range[0]) & (df_adult["age"] <= age_range[1]) &
    (df_adult["education"].isin(education_levels)) &
    (df_adult["income"] == income_level)
]

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_df)

# Data Visualization
st.header("Data Visualization")

# Bar chart of education levels
st.subheader("Education Levels Count")
edu_counts = filtered_df["education"].value_counts()
st.bar_chart(edu_counts)

# Histogram of ages
st.subheader("Age Distribution")
st.hist(filtered_df["age"], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Count')
st.pyplot()

# Box plot of hours per week by education
st.subheader("Box Plot of Hours Per Week by Education")
plt.figure(figsize=(10, 6))
sns.boxplot(x="education", y="hours-per-week", data=filtered_df)
plt.xticks(rotation=45)
plt.xlabel('Education Level')
plt.ylabel('Hours Per Week')
st.pyplot()

# Summary Statistics
st.header("Summary Statistics")
st.write(filtered_df.describe())

# Export filtered data to CSV
if st.button("Export Filtered Data to CSV"):
    filtered_df.to_csv("filtered_data.csv", index=False)
    st.success("Data exported successfully!")

# Footer
st.sidebar.text("Built with Streamlit by Your Name")

# Run the app
if __name__ == "__main__":
    st.run()
