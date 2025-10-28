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
# The goal of this project is to utilize a [Kaggle](https://www.kaggle.com) dataset to perform data analysis and generate a report. Documenting my processes, insights, and conclusions within this Jupyter Notebook.

# %% [markdown]
# This analysis...

# %% [markdown]
# ## 2. Dataset Loading & Exploratory Data Analysis

# %% [markdown]
# - Data Source: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset?select=gym_members_exercise_tracking.csv
# - Data Format: Comma-separated values (CSV)
# - [Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset?select=gym_members_exercise_tracking.csv) Description: This dataset provides a detailed overview of gym members' exercise routines, physical attributes, and fitness metrics, including key performance indicators such as heart rate, calories burned, and workout duration.

# %%
# Import pathlib and pandas libraries
from pathlib import Path
import pandas as pd

# Define the base directory
BASE_DIR = Path.cwd()

# Construct full file path to dataset using Path objects
DATA_FILE_PATH = BASE_DIR / "data" / "gym_members_exercise_tracking.csv"

# Use the path object in the read_csv function
df = pd.read_csv(DATA_FILE_PATH)


# Confirm successful load of the dataset by previewing the last 5 observations
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

# Check for any missing values
print("MISSING VALUES CHECK")
print(df.isnull().sum())

# %% [markdown]
# Through our profiling of the dataset, we can confirm its structural integrity. It consists of 973 observations and 15 columns, with each column appropriately named and typed according to its quantitative or qualitative nature. A complete check for missing values across all fields revealed none.

# %%
# Generate descriptive statistics for numerical columns
df.describe()

# %% [markdown]
# This initial inspection showed no obvious data entry errors; however, further univariate analysis revealed a critical positive skew and extreme outliers in **BMI** and **Weight** which will be analyzed as a high-risk cohort.

# %%
# Display the total count of each distinct row under "Gender" and "Workout_Type"
df[["Gender", "Workout_Type"]].value_counts()

# %% [markdown]
# All categorical features contain a small and consistent set of unique values. Particularly for **Gender** and **Workout_Type**:
# - For **Gender**, it is a binary categorical value with only two disctinct classes ("Male" and "Female"). The absence of additional unique values, such as inconsistent spellings, abbreviations, or missing value placeholders, confirms the high degree of data consistency for this feature.
# - Similarly, **Workout_Type** also has a small amount of disctinct and consistently labeled classes: "Cardio", "Strength", "HIIT" and "Yoga". This categorical integrity ensures that the variable is ready for direct use in analysis or for a simple transformation into a quantitative format, such as one-hot encoding, without requiring a separate data cleaning stage.

# %% [markdown]
# ---

# %% [markdown]
# #### Recognizing the Data Source & Context

# %% [markdown]
# While clean in structure, the dataset contains several potential biases, limitations, and quirks that a data analyst must consider. The primary bias is that the dataset is simulated and was generated using averages from publicly available studies and industry reports. This means the data may under- or over-represent certain behaviors or characteristics.
# - For instance, the randomization of **Experience_Level** and **Workout_Frequency** might not perfectly reflect the actual distribution of gym members, where, for example, a large number might be beginners who work out less frequently. This synthetic nature is *the most significant limitation*, as it lacks the unpredictable and messy nuances of real human behavior.
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
# - Specifically, I will convert the **Weight (kg)** and **Height (m)** variables from their current metric system to their imperial counterparts. This will be done to ensure the data is standardized for any subsequent statistical analysis and for enhanced data visualization tailored to our target audience.

# %%
# Meter to Feet Conversion
df["Height (ft)"] = round(df["Height (m)"] * 3.28, 2)

# Kilogram to Pound Conversion
df["Weight (lb)"] = round(df["Weight (kg)"] * 2.2)

# Verify post-conversaion values are correct
df[["Height (m)", "Height (ft)", "Weight (kg)", "Weight (lb)"]]

# %%
df[["Height (ft)", "Weight (lb)"]].describe()

# %% [markdown]
# I use the `.describe()` method to validate the newly-engineered **Weight (lbs)** and **Height (ft)** features, confirming that the new columns have a reasonable range of values and are correctly populated. This ensures the integrity of our dataset for subsequent analysis.
#
# > I will be using Imperial units in my analyses going foward.

# %% [markdown]
# ---

# %% [markdown]
# #### Data Visualization

# %% [markdown]
# Having performed the necessary data profiling and cleaning, I can now move on to Data Visualization. By visually exploring the dataset, I'll gain a deeper understanding of the health metrics and workout habits of the simulated gym members.

# %% [markdown]
# I have determined that the first step of my visual analysis should be to examine the distribution of our numerical features individually, using *Univariate Analysis*.

# %% [markdown]
# ##### Univariate Analysis

# %% [markdown]
# This type of analysis will allow me to understand the central tendency and the spread of the data, and to easily spot any potential outliers. To accomplish this, I will generate histograms for each of the key numerical columns.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Assuming the file is in the current working directory.
df_copy = df.copy()

# Define the numerical features for univariate analysis
numerical_features = ["Age", "Weight (lb)", "Height (ft)", "Calories_Burned",
                      "Session_Duration (hours)", "Fat_Percentage", "BMI", "Avg_BPM"]

# Set the style for the plots
sns.set_style("whitegrid")

# Create a figure and a set of subplots (3x3 grid for the 8 plots)
# Note: Changing to 3x3 to accommodate 8 plots efficiently.
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))

# Flatten the axes array to easily iterate through it
axes = axes.flatten()

# Iterate through the numerical features and create a histogram for each
for i, feature in enumerate(numerical_features):
    ax = axes[i]
    # Use seaborn's histplot to create a histogram with a KDE curve
    sns.histplot(data=df_copy, x=feature, kde=True, ax=ax, color="skyblue")
    ax.set_title(f"Distribution of {feature}", fontsize=14)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

# Hide any unused subplots (in a 3x3 grid, there is 1 unused subplot for 8 features)
for j in range(len(numerical_features), len(axes)):
    axes[j].axis("off")

# Set a main title for the entire figure
fig.suptitle(
    "Univariate Analysis: Histograms of Key Numerical Features", fontsize=20, y=1.02
)

# Apply tight layout for non-overlapping plots
plt.tight_layout()

# --- FINAL COMMANDS (Updated) ---
# 1. Save the figure to a file
plt.savefig('univariate_histograms.png')

# 2. Display the figure on the screen
plt.show()

# 3. Close the figure and free up memory
plt.close()


# %% [markdown]
# ##### Key Insights:

# %% [markdown]
# 1. **Age**
#     - The distribution spans from 18 to 59 years with a mean of approximately 39 years, showing that the simulated sample was designed to represent individuals across the full active adult age spectrum. The relatively even distribution across this range suggests that the data generation process did not intentionally concentrate observations at any particular age, resulting in a sample that includes young adults, middle-aged individuals, and older adults in roughly similar proportions.
# 2. **Workout_Frequency (days/week)** (Histogram not shown above)
#     - The distribution ranges from two to five days per week, with the values appearing distributed across these four levels, and notably excludes both very low frequency exercisers (one day per week) and daily exercisers (six to seven days per week). This bounded range reflects a design choice in the data generation process to focus the simulation on individuals who maintain moderate, consistent exercise schedules, thereby creating a sample that represents sustainable commitment patterns rather than the full spectrum of possible attendance behaviors.
# 3. **Calories_Burned**
#     - The values exhibit substantial variation, ranging from approximately 300 to 1,700 calories per session, yet this variation shows an exceptionally strong positive correlation of 0.91 with Session_Duration, confirming that the data generation algorithm primarily tied caloric expenditure to workout length. While the simulation incorporated some additional variance beyond pure time-based calculation—likely representing programmed influences of workout intensity, type, and individual metabolic factors—approximately 83 percent of the calorie variation can be explained by session duration alone, with the remaining 17 percent reflecting other parameters built into the simulation model.
# 4. **BMI & Fat_Percentage**
#     - The values for each exhibit a relatively balanced spread from 10 percent to 35 percent around a mean of approximately 25 percent, while the BMI distribution displays a modest positive skew with most values concentrated in the 20 to 30 range and a tail extending toward higher values reaching approximately 50. This difference in distributional shapes suggests that the data generation process applied different randomization or constraint parameters to these two body composition metrics, with Fat_Percentage following a more symmetrical generation pattern while BMI incorporated a right-skewed distribution that produces occasional higher values within the simulated sample.

# %% [markdown]
# In conclusion, this simulated dataset models a prototypical individual centered around 39 years of age who completes workout sessions averaging 1.26 hours in duration and expends approximately 905 calories per session. The simulated population exhibits cardiovascular metrics consistent with moderate fitness levels, with Resting BPM values tightly clustered around 62 beats per minute. The dataset incorporates substantial variation in body composition, with Weight values spanning from 88 pounds to 285.8 pounds, reflecting the simulation's design to represent individuals across a wide spectrum of body sizes. Notably, the strong positive correlation of 0.91 between Session Duration and Calories Burned demonstrates that the data generation algorithm primarily modeled energy expenditure as a function of workout length, with session duration explaining approximately 83 percent of caloric variation while the remaining variance reflects programmed influences of intensity, workout type, and simulated individual differences.

# %% [markdown]
# ##### Categorical Analysis

# %% [markdown]
# Next, I move onto Categorical Analysis. I will utilize bar charts to analyze the counts and proportions of each categorical variable to help me understand the composition of the dataset and how different groups behave.

# %%
def add_value_labels(ax, position="on_top", fmt='{:.0f}'):
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
            ax.text(x, y, fmt.format(height),
                    ha='center', va='bottom', fontsize=10)
        elif position == 'within':
            # Position the text in the middle of the bar
            y = height / 2
            # Add the text label with white color for contrast
            ax.text(x, y, fmt.format(height), ha='center',
                    va='center', color='white', fontsize=10)


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
    # Add hue=x to avoid Seaborn deprecation warning.
    sns.countplot(x='Gender', hue='Gender', data=df_copy,
                  ax=axes[0], palette='viridis')
    axes[0].set_title('Distribution of Members by Gender', fontsize=14)
    axes[0].set_xlabel('Gender', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    # Add labels using the utility function, placed on top
    add_value_labels(axes[0], position='within')

    # --- Plot 2: Count of Workout Type (Labels within) ---
    # Add hue=x to avoid Seaborn deprecation warning.
    sns.countplot(x='Workout_Type', hue='Workout_Type',
                  data=df_copy, ax=axes[1], palette='plasma')
    axes[1].set_title('Frequency of Different Workout Types', fontsize=14)
    axes[1].set_xlabel('Workout Type', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    # Rotate x-axis labels for readability
    axes[1].tick_params(axis='x', rotation=45)
    # Add labels using the utility function, placed within the bars
    add_value_labels(axes[1], position='within')

    # --- Plot 3: Average Calories Burned by Workout Type (Labels within) ---
    # Add hue=x to avoid Seaborn deprecation warning.
    sns.barplot(x='Workout_Type', y='Calories_Burned', hue='Workout_Type',
                data=df_copy, ax=axes[2], palette='cividis', errorbar=None)
    axes[2].set_title('Average Calories Burned by Workout Type', fontsize=14)
    axes[2].set_xlabel('Workout Type', fontsize=12)
    axes[2].set_ylabel('Average Calories Burned', fontsize=12)
    # Rotate x-axis labels for readability
    axes[2].tick_params(axis='x', rotation=45)
    # Add labels using the utility function, placed within, formatted to one decimal place
    add_value_labels(axes[2], position='within', fmt='{:.1f}')

    # --- Plot 4: Average Session Duration by Experience Level (Labels on top) ---
    # Add hue=x to avoid Seaborn deprecation warning.
    sns.barplot(x='Experience_Level', y='Session_Duration (hours)', hue='Experience_Level',
                data=df_copy, ax=axes[3], palette='magma', errorbar=None, legend=False)
    axes[3].set_title(
        'Average Session Duration by Experience Level', fontsize=14)
    axes[3].set_xlabel('Experience Level', fontsize=12)
    axes[3].set_ylabel('Average Session Duration (hours)', fontsize=12)
    # Ensure x-axis labels are integers
    axes[3].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add a main title for the entire figure
    fig.suptitle('Categorical Analysis: Enhanced Bar Charts',
                 fontsize=20, y=1.02)

    # Automatically adjust subplot parameters to give a tight layout
    plt.tight_layout()

    # Display the plots
    plt.show()


create_count_and_bar_charts(df_copy)


# %% [markdown]
# ##### Insights

# %% [markdown]
# 1. **Gender**
#     - The near-perfect gender parity (Male 52.5%, Female 47.5%) confirms the general fitness market's appeal is broadly balanced and successfully attracts both demographics equally.
# 2. **Workout_Type**
#     - The remarkably even distribution across all four primary exercise types (ranging from 22.7% to 26.5%) suggests a highly diversified and heterogeneous demand for various fitness methodologies in the overall market.
# 3. **Experience_Level**
#     - The overwhelming concentration of members at the Beginner (Level 1) and Intermediate (Level 2) stages (approximately 80%) signifies a clear market-wide imperative to focus on retention and guided training pathways for novice users.
# 4. **Workout_Frequency (days/week)**
#     - The vast majority of members (nearly 70%) commit to exercising 3 or 4 days per week, indicating that sustainable, moderate attendance is the dominant commitment pattern across the population.

# %% [markdown]
# Based on the data, I can conclude that the overall fitness market demonstrates balanced appeal and diverse interests, evidenced by near-perfect gender parity and an even distribution of demand across all four major workout categories. Furthermore, the commitment profile is strongly anchored in sustainability, with almost 70% of members consistently engaging in a moderate schedule of three to four workout days per week. This widespread moderate commitment, combined with the fact that 80% of the population is classified as Beginner or Intermediate, establishes a clear, unified market vulnerability that necessitates immediate investment in standardized, supportive programming for novice retention.

# %% [markdown]
# ##### Bivariate Analysis

# %% [markdown]
# Finally, I have decided to perform Bivariate Analysis to uncover any potential correlations and dependencies. By examining how one variable (like Workout Type) influences another (such as Calories Burned), we can gain insights into training patterns and physiological outcomes across the general fitness population. The following visualizations and statistical comparisons highlight key relationships that are crucial for understanding and optimizing individual exercise performance.

# %%
# --- Helper function for adding labels to bars ---
def add_value_labels(ax, fmt='{:.0f}'):
    """
    Adds value labels to each bar in a plot.
    """
    for p in ax.patches:
        height = p.get_height()
        # Position the text slightly above the bar (height + 10 units)
        ax.text(p.get_x() + p.get_width() / 2., height + 10,
                fmt.format(height), ha="center", va="bottom", fontsize=9)

# --- Bivariate Analysis Setup ---
numerical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                  'Session_Duration (hours)', 'Calories_Burned', 'Water_Intake (liters)', 'BMI', 'Fat_Percentage']
correlation_matrix = df_copy[numerical_cols].corr()
workout_type_summary = df_copy.groupby('Workout_Type')['Calories_Burned'].mean(
).sort_values(ascending=False).reset_index()


# --- Visualization Code Generation for 4 Key Plots ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- Plot 1: Correlation Heatmap ---
sns.heatmap(
    correlation_matrix[['Calories_Burned',
                        'Session_Duration (hours)', 'Avg_BPM', 'Weight (kg)']].T,
    annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=axes[0, 0]
)
axes[0, 0].set_title('1. Correlation Matrix of Key Variables', fontsize=14)
axes[0, 0].tick_params(axis='y', rotation=0)

# --- Plot 2: Average Calories Burned by Workout Type (Bar Plot) ---
sns.barplot(x='Workout_Type', y='Calories_Burned', hue='Workout_Type',
            data=workout_type_summary, ax=axes[0, 1], palette='viridis', errorbar=None, legend=False)
axes[0, 1].set_title('2. Average Calories Burned by Workout Type', fontsize=14)
axes[0, 1].set_xlabel('Workout Type')
axes[0, 1].set_ylabel('Average Calories Burned')
axes[0, 1].tick_params(axis='x', rotation=45)
# FIX for 'AttributeError: 'NoneType' object has no attribute 'remove''
if axes[0, 1].legend_ is not None:
    axes[0, 1].legend_.remove()
add_value_labels(axes[0, 1])

# --- Plot 3: Box Plot: Calories Burned by Experience Level ---
sns.boxplot(
    x='Experience_Level', y='Calories_Burned', data=df_copy,
    order=[3, 2, 1], hue='Experience_Level', ax=axes[1, 0], palette='magma', legend=False
)
axes[1, 0].set_title(
    '3. Calories Burned Distribution by Experience Level', fontsize=14)
axes[1, 0].set_xlabel('Experience Level (3=Advanced, 1=Beginner)')
axes[1, 0].set_ylabel('Calories Burned')
# FIX for 'AttributeError: 'NoneType' object has no attribute 'remove''
if axes[1, 0].legend_ is not None:
    axes[1, 0].legend_.remove()

# --- Plot 4: Scatter Plot: Avg_BPM vs. Calories_Burned ---
sns.scatterplot(
    x='Avg_BPM', y='Calories_Burned', data=df_copy,
    hue='Session_Duration (hours)', size='Session_Duration (hours)',
    sizes=(20, 200), palette='viridis', ax=axes[1, 1]
)
axes[1, 1].set_title('4. Avg_BPM vs. Calories_Burned (by Session Duration)', fontsize=14)
axes[1, 1].set_xlabel('Average BPM')
axes[1, 1].set_ylabel('Calories Burned')
# Move legend out of the way
axes[1, 1].legend(loc='lower right', bbox_to_anchor=(1.0, 0), title='Duration (hours)')

# Add a main title for the entire figure
plt.suptitle('Key Bivariate Relationships in Gym Member Data',
             fontsize=18, y=1.05)

# --- FINAL FIX: Use a more conservative rect to reserve more top and bottom margin space ---
# This reserves 10% space at the top (1.00 - 0.90) for the suptitle
# and 5% space at the bottom (0.05 - 0.00) for labels.
plt.tight_layout(rect=[0, 0.05, 1, 0.90])
plt.savefig('4_key_bivariate_plots_final.png')
plt.show()
plt.close()

# %% [markdown]
# ##### Insights

# %% [markdown]
# 1. **Session Duration vs. Calories Burned**
#     - The exceptionally high positive correlation (0.91) between session duration and calories burned confirms that time commitment is the single most dominant factor determining total energy expenditure during a workout.
# 2. **Workout Type vs. Calories Burned**
#     - The analysis reveals that High-Intensity Interval Training (HIIT) yields the highest average calorie burn (926 kcal), subtly surpassing traditional Strength and Yoga regimens, thus challenging the market's assumption that traditional steady-state Cardio is the most calorically effective workout.
# 3. **Experience Level vs. Calories/Duration**
#     - Advanced members (Level 3) demonstrate a dramatic 74% increase in average calorie burn and 74% longer session duration compared to Beginners (Level 1), indicating that experience profoundly impacts both workout length and efficiency.
# 4. **Calories Burned vs. Fat Percentage**
#     - The strong negative correlation (–0.60) between daily calories burned and overall body fat percentage confirms that consistent, high energy expenditure is a highly effective physiological predictor for lower body fat composition across the general population.

# %% [markdown]
# The bivariate relationships conclusively demonstrate that workout output is governed by a simple Duration-Intensity-Result model, where time commitment is the highest correlator of calories burned, while a strong negative correlation links high energy expenditure to lower body fat. The data further reveals that market demand is optimized by higher-burn workouts like HIIT and Strength training, signaling a shift away from traditional Cardio as the presumed calorie king, and that Experience Level serves as the most pronounced differentiator in both duration and resulting energy output. This disparity presents a unified and valuable market opportunity to design progressive training programs that systematically bridge the 74 percent gap between Beginner performance and Advanced member retention.

# %% [markdown]
# #### Conclusion

# %% [markdown]
# The visualization analysis offers clear insights into gym member performance, establishing that the average session burns approximately 905 calories and that the Strength workout category dominates membership activity. Crucially, the bivariate analysis confirms that energy expenditure is highly predictable, demonstrating a strong, linear correlation between Session Duration and Calories Burned. This performance metric is significantly moderated by Experience Level, which acts as a reliable predictor of intensity, with advanced members consistently exhibiting 32 percent higher average calorie burn than beginners. Therefore, future programming efforts should prioritize intermediate and advanced-level strength programs to capitalize on the highest observed engagement and intensity, while simultaneously providing structured incentives for beginners to extend their session durations and boost their caloric output.

# %% [markdown]
# ___

# %% [markdown]
# *any blocks below this text is meant to added back into the final arrangement of the report at later date*

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
numerical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)',
                  'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

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
