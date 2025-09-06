# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project Title
# *all language subject to change*
#
# **Author:** Axel Christian Cabato
#
# **Date:** [Date]

# %% [markdown]
# # 1. Introduction
# The goal of this project is to utilize a [Kaggle](https://www.kaggle.com) dataset to perform Data Analysis and generate a report. Documenting my processes, insights, and conclusions within this Jupyter Notebook.

# %% [markdown]
# This analysis...

# %%
# Import pandas
import pandas as pd

# %% [markdown]
# ## 2. Dataset Loading & Exploratory Data Analysis

# %% [markdown]
# - Data Source: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset?select=gym_members_exercise_tracking.csv
# - Data Format: Comma-separated values (CSV)
# - [Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset?select=gym_members_exercise_tracking.csv) Description: This dataset provides a detailed overview of gym members' exercise routines, physical attributes, and fitness metrics, including key performance indicators such as heart rate, calories burned, and workout duration.

# %%
# Load the data into a pandas DataFrame using .read_csv()
df = pd.read_csv("C:/Users/AxelC/Desktop/ME/Career/PROJECTS/Data Science & Analysis/--First DA Project/gym_members_exercise_tracking.csv")

# Confirming successful load of the dataset by previewing the last 5 observations
df.tail()

# %% [markdown]
# ### Exploratory Data Analysis

# %% [markdown]
# #### Data Profiling

# %%
# Output a concicise summary of the DataFrame
print("DATAFRAME SUMMARY")
df.info()

print("\n")

# Check the dataset for any missing values
print("MISSING VALUES CHECK")
print(df.isnull().sum())

# %% [markdown]
# Through our profiling of the dataset, we can confirm its structural integrity. It consists of 973 observations and 15 columns, with each column appropriately named and typed according to its quantitative or qualitative nature. A complete check for missing values across all fields revealed none. Also, this initial inspection of the data showed no signs of extreme outliers or data entry errors.

# %%
# Generate descriptive statistics for "BMI" & "Fat_Percentage" columns
df.describe()

# %% [markdown]
# All numerical features appear to have a reasonable range of values.

# %% [markdown]
# For **BMI** and **Fat_Percentage**: 
# - The minimum and maximum values for both are physiologically plausible, confirming the data is within a realistic human-centric scale.
# - They also exhibit a symmetrical distribution, as evidenced by their mean and median values being in close proximity. 
#     - This central tendency, combined with a reasonable standard deviation for each, suggests the absence of significant outliers, indicating a consistent and predictable spread of values for both metrics within the dataset.

# %%
# Return the frequency of each distinct row in the DataFrame
df[["Gender", "Workout_Type"]].value_counts()

# %% [markdown]
# All categorical features contain a small and consistent set of unique values. Particularly for **Gender** and **Workout_Type**:
# - For **Gender**, it is a binary categorical value with only two disctinct classes ("Male" and "Female"). The absence of additional unique values, such as inconsistent spellings, abbreviations, or missing value placeholders, confirms the high degree of data consistency for this feature.
# - Similarly, **Workout_Type** also has two disctinct and consistently labeled classes: "Cardio" and "Strength". This categorical integrity ensures that the variable is ready for direct use in analysis or for a simple transformation into a quantitative format, such as one-hot encoding, without requiring a separate data cleaning stage.

# %% [markdown]
# #### Recognizing the Data Source & Context

# %% [markdown]
# While clean in structure, the dataset contains several potential biases, limitations, and quirks that a data analyst must consider. The primary bias is that the dataset is simulated and was generated using averages from publicly available studies and industry reports. This means the data may under- or over-represent certain behaviors or characteristics.
# - For instance, the randomization of Experience_Level and Workout_Frequency might not perfectly reflect the actual distribution of gym members, where, for example, a large number might be beginners who work out less frequently. **This synthetic nature is the most significant limitation, as it lacks the unpredictable and messy nuances of real human behavior.**
# - Any insights or models derived from this dataset would need to be validated with actual, real-world data before being applied to a genuine scenario.
#
# The dataset also has a few quirks that are uncommon in real-world data. It has **no missing values** and all categorical values are perfectly consistent, *which is highly unusual*. 
#
# Furthermore, the data is simplified and contains only the variables that were explicitly defined in the generation process. 
# - For example, the **Workout_Type** column is limited to a small, consistent set of categories (*Cardio*, *Strength*, *Yoga*, *HIIT*), and does not reflect the full range of possible exercises performed by gym members.

# %% [markdown]
# > This foundational understanding will serve as a solid basis for our deeper exploratory data analysis.

# %% [markdown]
# ---

# %% [markdown]
# To prepare the data for any audience who might be more familiar with imperial units, I will perform some feature engineering by constructing new attributes from the existing dataset.
# - Specifically, I will convert the `Weight (kg)` and `Height (m)` variables from their current metric system to their imperial counterparts. This will be done to ensure the data is standardized for any subsequent statistical analysis and for enhanced data visualization tailored to our target audience.

# %%
df["Height (ft)"] = round(df["Height (m)"] * 3.28, 2)
df["Weight (lb)"] = round(df["Weight (kg)"] * 2.2)

df[["Height (m)", "Height (ft)","Weight (kg)", "Weight (lb)"]]

# %%
df[["Height (ft)", "Weight (lb)"]].describe()

# %% [markdown]
# ---

# %% [markdown]
# #### Data Visualization

# %% [markdown]
# ## 3. Data Cleaning & Transformation

# %% [markdown]
# Since the dataset has no missing values, inconsistent formats, or clear outliers, we conclude it is "clean" and does not need to undergo any data cleansing.

# %% [markdown]
# ### Data Transformation

# %% [markdown]
# This section outlines the process of transforming the raw dataset to ensure it is in the optimal format for analysis and model building. While the dataset has high structural integrity with no missing values, several features require transformation to be used effectively. Specifically, we will address the conversion of categorical data into a numerical format, and the scaling of numerical features to standardize their range. This process ensures all variables are ready for potential use in machine learning models, preventing potential issues with feature bias and performance.

# %% [markdown]
# #### Feature Scaling

# %% [markdown]
# We standardize the numerical columns using `StandardScaler` from the `scikit-learn` library, transforming its values so that they have a **mean of 0** and a **standard deviation of 1**.

# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Identify numerical columns to scale
numerical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the numerical data
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the scaled data (first 5 rows)
print(df[numerical_cols].head())

# %% [markdown]
# #### Encoding Categorical Variables

# %% [markdown]
# We use `get_dummies()` from `pandas` to perform one-hot encoding.

# %%
# Identify categorical columns to encode
categorical_cols = ['Gender', 'Workout_Type']

# Perform one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Display the transformed DataFrame with the new columns
print(df.head())

# %% [markdown]
# #### Binning/Discretization

# %% [markdown]
# #### Feature Engineering

# %% [markdown]
# #### Skewed Data Handling

# %% [markdown]
# #### Dimensionality Reduction

# %% [markdown]
# ---

# %% [markdown]
# *any blocks below this text is meant to added back into the final arrangement of the report at later date*

# %%
# Import visualization libraries, Matplotlib & seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.histplot(data=df, x="Age", stat="count").set(title="Age Histogram Plot")

# %% [markdown]
# Based on the histogram, we have fairly distributed range of observed individuals
