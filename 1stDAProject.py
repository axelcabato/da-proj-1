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
df = pd.read_csv("C:/Users/AxelC/Desktop/ME/Career/PROJECTS/Data Science & Analysis/da-proj-1-1/gym_members_exercise_tracking.csv")

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
# I use the `.describe()` method to validate the newly-engineered `Weight (lbs)` and `Height (ft)` features, confirming that the new columns have a reasonable range of values and are correctly populated. This ensures the integrity of our dataset for subsequent analysis.
#
# > I will be using Imperial units in my analyses going foward.

# %% [markdown]
# ---

# %% [markdown]
# #### Data Visualization

# %% [markdown]
# Having performed the necessary data profiling and cleaning, I can now move on to Data Visualization. By visually exploring the dataset, I'll gain a deeper understanding of the health metrics and workout habits of the simulated gym members.

# %% [markdown]
# Firstly, I have determined that the first step of my visual analysis should be to examine the distribution of our numerical features individually. 

# %% [markdown]
# **Univariate analysis** will allow me to understand the central tendency, the spread of the data, and to easily spot any potential outliers. To accomplish this, I will generate histograms for each of the key numerical columns.

# %%
# Import Matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Define the numerical features for univariate analysis
numerical_features = ['Age', 'Weight (lb)', 'Height (ft)', 'Calories_Burned', 'Session_Duration (hours)', 'Fat_Percentage', 'BMI']

# Set the style for the plots
sns.set_style("whitegrid")

# Create a figure and a set of subplots
# We will use a 3x3 grid for the 7 plots.
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
# Flatten the axes array to easily iterate through it
axes = axes.flatten()

# Iterate through the numerical features and create a histogram for each
for i, feature in enumerate(numerical_features):
    ax = axes[i]
    # Use seaborn's histplot to create a histogram with a KDE curve
    sns.histplot(data=df, x=feature, kde=True, ax=ax, color='skyblue')
    ax.set_title(f'Distribution of {feature}', fontsize=14)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

# Hide any unused subplots
for j in range(len(numerical_features), len(axes)):
    axes[j].axis('off')

# Set a main title for the entire figure
fig.suptitle('Univariate Analysis: Histograms of Key Numerical Features', fontsize=20, y=1.02)

# Adjust layout to prevent titles from overlapping
plt.tight_layout()

# Display the plots
plt.show()


# %% [markdown]
# Next, I shift my focus onto the categorical data. I will use bar charts to compare a numerical variable's summary statistics across different categories, or simply to count how many members fall into each category. This helps me understand the composition of our dataset and how different groups behave.

# %%
def add_value_labels(ax, position='on_top', fmt='{:.0f}'):
    """
    Adds value labels to each bar in a plot.

    Args:
        ax (plt.Axes): The axes object to add labels to.
        position (str): The position of the labels.
                        'on_top' (default) places labels above the bars.
                        'within' places labels inside the bars.
        fmt (str): The format string for the labels (e.g., '{:.0f}' for integers,
                   '{:.1f}' for one decimal place).
    """
    # Loop over each bar (patch) in the axes
    for p in ax.patches:
        # Get the height of the bar
        height = p.get_height()
        # Define the x-coordinate for the text (center of the bar)
        x = p.get_x() + p.get_width() / 2.
        
        if position == 'on_top':
            # Position the text slightly above the bar
            y = height + 1
            # Add the text label
            ax.text(x, y, fmt.format(height), ha='center', va='bottom', fontsize=10)
        elif position == 'within':
            # Position the text in the middle of the bar
            y = height / 2
            # Add the text label with white color for contrast
            ax.text(x, y, fmt.format(height), ha='center', va='center', color='white', fontsize=10)

def create_count_and_bar_charts(df):
    """
    Generates a set of categorical charts with labels placed on top of or within the bars.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data for the plots.
    """
    # Set the visual style for the plots using seaborn
    sns.set_style("whitegrid")
    
    # Create a figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    # Flatten the axes array to easily iterate through them
    axes = axes.flatten()

    # --- Plot 1: Count of Gender (Labels on top) ---
    # Add hue=x to avoid the Seaborn deprecation warning.
    sns.countplot(x='Gender', hue='Gender', data=df, ax=axes[0], palette='viridis')
    axes[0].set_title('Distribution of Members by Gender', fontsize=14)
    axes[0].set_xlabel('Gender', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    # Add labels using the utility function, placed on top
    add_value_labels(axes[0], position='within')

    # --- Plot 2: Count of Workout Type (Labels within) ---
    # Add hue=x to avoid the Seaborn deprecation warning.
    sns.countplot(x='Workout_Type', hue='Workout_Type', data=df, ax=axes[1], palette='plasma')
    axes[1].set_title('Frequency of Different Workout Types', fontsize=14)
    axes[1].set_xlabel('Workout Type', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    # Rotate x-axis labels for readability
    axes[1].tick_params(axis='x', rotation=45)
    # Add labels using the utility function, placed within the bars
    add_value_labels(axes[1], position='within')
    
    # --- Plot 3: Average Calories Burned by Workout Type (Labels within) ---
    # Add hue=x to avoid the Seaborn deprecation warning.
    sns.barplot(x='Workout_Type', y='Calories_Burned', hue='Workout_Type', data=df, ax=axes[2], palette='cividis', errorbar=None)
    axes[2].set_title('Average Calories Burned by Workout Type', fontsize=14)
    axes[2].set_xlabel('Workout Type', fontsize=12)
    axes[2].set_ylabel('Average Calories Burned', fontsize=12)
    # Rotate x-axis labels for readability
    axes[2].tick_params(axis='x', rotation=45)
    # Add labels using the utility function, placed within, formatted to one decimal place
    add_value_labels(axes[2], position='within', fmt='{:.1f}')
    
    # --- Plot 4: Average Session Duration by Experience Level (Labels on top) ---
    # Add hue=x to avoid the Seaborn deprecation warning.
    sns.barplot(x='Experience_Level', y='Session_Duration (hours)', hue='Experience_Level', data=df, ax=axes[3], palette='magma', errorbar=None, legend=False)
    axes[3].set_title('Average Session Duration by Experience Level', fontsize=14)
    axes[3].set_xlabel('Experience Level', fontsize=12)
    axes[3].set_ylabel('Average Session Duration (hours)', fontsize=12)
    # Ensure x-axis labels are integers
    axes[3].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add a main title for the entire figure
    fig.suptitle('Categorical Analysis: Enhanced Bar Charts', fontsize=20, y=1.02)

    # Automatically adjust subplot parameters to give a tight layout
    plt.tight_layout()

    # Display the plots
    plt.show()

create_count_and_bar_charts(df)

# %% [markdown]
# ___

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
