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
# This initial inspection showed no obvious data entry errors; however, further univariate analysis will reveal a critical positive skew and extreme outliers in `BMI` and `Weight` which will be analyzed as a high-risk cohort.

# %%
# Display the total count of each distinct row under "Gender" and "Workout_Type"
df[["Gender", "Workout_Type"]].value_counts()

# %% [markdown]
# The categorical features of `Gender` and `Workout_Type` contain a small and consistent set of unique values.:
# - For `Gender`, it is a binary categorical value with only two disctinct classes ("Male" and "Female"). The absence of additional unique values, such as inconsistent spellings, abbreviations, or missing value placeholders, confirms the high degree of data consistency for this feature.
# - Similarly, `Workout_Type` also has a small amount of disctinct and consistently labeled classes: "Cardio", "Strength", "HIIT" and "Yoga". This categorical integrity ensures that the variable is ready for direct use in analysis or for a simple transformation into a quantitative format, such as one-hot encoding, without requiring a separate data cleaning stage.

# %% [markdown]
# ---

# %% [markdown]
# #### Recognizing the Data Source & Context

# %% [markdown]
# While clean in structure, the dataset contains several potential biases, limitations, and quirks that a data analyst must consider. The primary bias is that the dataset was simulated and generated using averages from publicly available studies and industry reports. This means the data may under- or over-represent certain behaviors or characteristics.
# - For instance, the randomization of `Experience_Level` and `Workout_Frequency` might not perfectly reflect the actual distribution of gym members, where, for example, a large number might be beginners who work out less frequently. This synthetic nature is *the most significant limitation*, as it lacks the unpredictable and messy nuances of real human behavior.
#
# **Any insights or models derived from this dataset would need to be validated with actual, real-world data before being applied to a genuine scenario.**
#
# The dataset also has a few quirks that are uncommon in real-world data:
# - It **has no missing values** and **all categorical values are perfectly consistent**, *which is highly unusual*.
#
# Furthermore, the data is simplified and contains only the variables that were explicitly defined in the generation process. 
# - For example, the `Workout_Type` column is limited to a small, consistent set of categories (*Cardio*, *Strength*, *Yoga*, *HIIT*), and does not reflect the full range of possible exercises performed by gym members.

# %% [markdown]
# > This foundational understanding will serve as a solid basis for our deeper exploratory data analysis of this dataset.

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
# I use the `.describe()` method to validate the newly-engineered Weight (lbs)** and **Height (ft)** features, confirming that the new columns have a reasonable range of values and are correctly populated. This ensures the integrity of our dataset for subsequent analysis.
#
# > I will be using Imperial units in my analyses going foward.

# %% [markdown]
# ---

# %% [markdown]
# #### Data Visualization

# %% [markdown]
# Having performed the necessary data profiling and cleaning, I can now move on to visually exploring the dataset. By doing so I'll gain a deeper understanding of the health metrics and workout habits of the simulated gym members.

# %% [markdown]
# My first step of my visual analysis should be to examine the distribution of our numerical features individually, using *Univariate Analysis*.

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
                      "Session_Duration (hours)", "Fat_Percentage", "BMI", "Avg_BPM", "Resting_BPM"]

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
# 1. `Age`
#     - The distribution spans from 18 to 59 years with a mean of approximately 39 years, showing that the simulated sample was designed to represent individuals across the full active adult age spectrum. The relatively even distribution across this range suggests that the data generation process did not intentionally concentrate observations at any particular age, resulting in a sample that includes young adults, middle-aged individuals, and older adults in roughly similar proportions.
# 2. `Workout_Frequency (days/week)` (Histogram not shown above)
#     - The distribution ranges from two to five days per week, with the values appearing distributed across these four levels, and notably excludes both very low frequency exercisers (one day per week) and daily exercisers (six to seven days per week). This bounded range reflects a design choice in the data generation process to focus the simulation on individuals who maintain moderate, consistent exercise schedules, thereby creating a sample that represents sustainable commitment patterns rather than the full spectrum of possible attendance behaviors.
# 3. `Calories_Burned`
#     - The values exhibit substantial variation, ranging from approximately 300 to 1,700 calories per session, yet this variation shows an exceptionally strong positive correlation of 0.91 with Session_Duration, confirming that the data generation algorithm primarily tied caloric expenditure to workout length. While the simulation incorporated some additional variance beyond pure time-based calculation—likely representing programmed influences of workout intensity, type, and individual metabolic factors—approximately 83 percent of the calorie variation can be explained by session duration alone, with the remaining 17 percent reflecting other parameters built into the simulation model.
# 4. `BMI` & `Fat_Percentage`
#     - The values for each exhibit a relatively balanced spread from 10 percent to 35 percent around a mean of approximately 25 percent, while the BMI distribution displays a modest positive skew with most values concentrated in the 20 to 30 range and a tail extending toward higher values reaching approximately 50. This difference in distributional shapes suggests that the data generation process applied different randomization or constraint parameters to these two body composition metrics, with Fat_Percentage following a more symmetrical generation pattern while BMI incorporated a right-skewed distribution that produces occasional higher values within the simulated sample.

# %% [markdown]
# In conclusion, this simulated dataset models a prototypical individual centered around 39 years of age who completes workout sessions averaging 1.26 hours in duration and expends approximately 905 calories per session. The simulated population exhibits cardiovascular metrics consistent with moderate fitness levels, with Resting BPM values tightly clustered around 62 beats per minute. The dataset incorporates substantial variation in body composition, with Weight values spanning from 88 pounds to 285.8 pounds, reflecting the simulation's design to represent individuals across a wide spectrum of body sizes. Notably, the strong positive correlation of 0.91 between Session Duration and Calories Burned demonstrates that the data generation algorithm primarily modeled energy expenditure as a function of workout length, with session duration explaining approximately 83 percent of caloric variation while the remaining variance reflects programmed influences of intensity, workout type, and simulated individual differences.

# %% [markdown]
# ##### Categorical Analysis

# %% [markdown]
# Next, I move onto Categorical Analysis. I will utilize bar charts to analyze the counts and proportions of each categorical variable to help me understand the composition of the dataset and how different groups behave.

# %%
# Set the visual style for all plots
sns.set_style("whitegrid")

# Create a figure with a 2x2 grid of subplots
# figsize=(16, 12) means 16 inches wide by 12 inches tall
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Flatten the 2D array of axes into a 1D array for easier iteration
axes = axes.flatten()

# Helper function to add labels inside bars
def add_value_labels_inside(ax):
    """
    Adds value labels inside each bar in a bar plot.
    
    Parameters:
    ax: The axes object containing the bar plot
    """
    # Loop through each bar (patch) in the axes
    for patch in ax.patches:
        # Get the height (value) of the bar
        height = patch.get_height()
        
        # Calculate the x-coordinate (center of the bar)
        x = patch.get_x() + patch.get_width() / 2
        
        # Calculate the y-coordinate (middle of the bar)
        y = height / 2
        
        # Add the text label in the center of the bar
        ax.text(
            x, y,                    # Position (x, y)
            f'{int(height)}',        # Text to display (the count)
            ha='center',             # Horizontal alignment: center
            va='center',             # Vertical alignment: center
            fontsize=11,             # Font size
            fontweight='bold',       # Make text bold
            color='white'            # White text for contrast against colored bars
        )

# --- PLOT 1: Gender Distribution ---
sns.countplot(
    data=df,
    x='Gender',
    hue='Gender',
    ax=axes[0],
    palette='viridis',
    legend=False
)
axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Gender', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)

# Add value labels inside the bars
add_value_labels_inside(axes[0])

# --- PLOT 2: Workout Type Distribution ---
sns.countplot(
    data=df,
    x='Workout_Type',
    hue='Workout_Type',
    ax=axes[1],
    palette='plasma',
    legend=False
)
axes[1].set_title('Workout Types Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Workout Type', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# Add value labels inside the bars
add_value_labels_inside(axes[1])

# --- PLOT 3: Experience Level Distribution ---
sns.countplot(
    data=df,
    x='Experience_Level',
    hue='Experience_Level',
    ax=axes[2],
    palette='magma',
    legend=False,
    order=[1, 2, 3]
)
axes[2].set_title('Experience Level Distribution', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Experience Level (1=Beginner, 2=Intermediate, 3=Advanced)', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)

# Add value labels inside the bars
add_value_labels_inside(axes[2])

# --- PLOT 4: Workout Frequency Distribution ---
sns.countplot(
    data=df,
    x='Workout_Frequency (days/week)',
    hue='Workout_Frequency (days/week)',
    ax=axes[3],
    palette='cividis',
    legend=False,
    order=[2, 3, 4, 5]
)
axes[3].set_title('Workout Frequency Distribution', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Workout Frequency (days/week)', fontsize=12)
axes[3].set_ylabel('Count', fontsize=12)

# Add value labels inside the bars
add_value_labels_inside(axes[3])

# Add a main title for the entire figure
fig.suptitle(
    'Categorical Analysis: Distribution of Key Features',
    fontsize=18,
    fontweight='bold',
    y=0.995
)

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Save the figure to a file
#plt.save

# %% [markdown]
# ##### Bivariate Analysis

# %% [markdown]
# Finally, I have decided to perform Bivariate Analysis to uncover potential correlations and dependencies within the dataset. By examining how one variable (like Workout Type) relates to another (such as Calories Burned), I can identify the mathematical relationships programmed into the data generation process. The following visualizations and statistical comparisons highlight key relationships embedded in the dataset structure.

# %%
# Set visual style
sns.set_style("whitegrid")

# Prepare data for specific visualizations
numerical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                  'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
                  'Water_Intake (liters)', 'BMI', 'Fat_Percentage']
correlation_matrix = df[numerical_cols].corr()
workout_type_summary = df.groupby('Workout_Type')['Calories_Burned'].mean().sort_values(ascending=False).reset_index()

# Create figure with 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Plot 1: Correlation Heatmap ---
sns.heatmap(
    correlation_matrix[['Calories_Burned', 'Session_Duration (hours)', 'Avg_BPM', 'Fat_Percentage']].T,
    annot=True, fmt='.2f', cmap='coolwarm', cbar=True, 
    linewidths=0.5, linecolor='black', ax=axes[0, 0]
)
axes[0, 0].set_title('Correlation Matrix: Key Variable Relationships', fontsize=14, fontweight='bold')
axes[0, 0].tick_params(axis='y', rotation=0)
axes[0, 0].tick_params(axis='x', rotation=45)

# --- Plot 2: Average Calories Burned by Workout Type ---
sns.barplot(
    data=workout_type_summary, x='Workout_Type', y='Calories_Burned',
    hue='Workout_Type', ax=axes[0, 1], palette='viridis', legend=False
)
axes[0, 1].set_title('Average Calories Burned by Workout Type', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Workout Type', fontsize=12)
axes[0, 1].set_ylabel('Average Calories Burned', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

# Add value labels inside bars
for patch in axes[0, 1].patches:
    height = patch.get_height()
    axes[0, 1].text(
        patch.get_x() + patch.get_width() / 2, height / 2,
        f'{int(height)}', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white'
    )

# --- Plot 3: Box Plot - Calories Burned by Experience Level ---
sns.boxplot(
    data=df, x='Experience_Level', y='Calories_Burned',
    order=[3, 2, 1], hue='Experience_Level', ax=axes[1, 0], 
    palette='magma', legend=False
)
axes[1, 0].set_title('Calories Burned Distribution by Experience Level', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Experience Level (3=Advanced, 2=Intermediate, 1=Beginner)', fontsize=12)
axes[1, 0].set_ylabel('Calories Burned', fontsize=12)

# --- Plot 4: Scatter Plot - Avg_BPM vs Calories_Burned ---
scatter = sns.scatterplot(
    data=df, x='Avg_BPM', y='Calories_Burned',
    hue='Session_Duration (hours)', size='Session_Duration (hours)',
    sizes=(20, 200), palette='viridis', ax=axes[1, 1], alpha=0.6
)
axes[1, 1].set_title('Average BPM vs Calories Burned (by Session Duration)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Average BPM', fontsize=12)
axes[1, 1].set_ylabel('Calories Burned', fontsize=12)
axes[1, 1].legend(title='Duration (hrs)', loc='lower right', fontsize=9)

# Add main title
fig.suptitle('Key Bivariate Relationships', 
             fontsize=18, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout()

# Save and display
plt.savefig('bivariate_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %% [markdown]
# ##### Key Insights
#
# 1. `Session_Duration` vs. `Calories_Burned`
#     - The correlation matrix reveals an exceptionally strong positive correlation of 0.91 between Session_Duration and Calories_Burned, confirming that the data generation algorithm primarily tied caloric expenditure to workout length. This relationship accounts for approximately 83 percent of the variance in calories burned, with the remaining 17 percent reflecting programmed influences of workout intensity, exercise type, and simulated individual metabolic differences.
#
# 2. `Workout_Type` vs. `Calories_Burned`
#     - The average calories burned across workout types shows HIIT at 926 calories, followed by Strength at 905 calories, Cardio at 903 calories, and Yoga at 901 calories. This relatively narrow range (25-calorie spread) indicates the simulation programmed minimal differentiation in caloric output between workout modalities, with all four types clustering tightly around the overall mean of 905 calories.
#
# 3. `Experience_Level` vs. `Calories_Burned`
#     - The box plot distribution reveals that Advanced members (Level 3) have median calorie burns approximately 400 calories higher than Beginners (Level 1), with Intermediate members (Level 2) falling between these extremes. This systematic progression reflects a design choice to model experience level as a strong predictor of workout output, with higher experience correlating with both longer session durations and greater caloric expenditure.
#
# 4. `Average_BPM` vs. `Calories_Burned`
#     - The scatter plot demonstrates a positive relationship between Average_BPM and Calories_Burned, with color gradation showing that longer session durations correspond to both higher heart rates and greater caloric expenditure. This pattern indicates the simulation modeled cardiovascular intensity as interconnected with both workout duration and energy expenditure, creating a three-way relationship where time, heart rate, and calories increase together.

# %% [markdown]
# In conclusion, the bivariate relationships within this simulated dataset demonstrate that the data generation algorithm was programmed with Session_Duration as the dominant predictor of caloric expenditure, accounting for 83 percent of the variance through a strong positive correlation of 0.91. The simulation incorporated secondary variation through Experience_Level, which shows systematic progression in energy output from beginner to advanced classifications, while Workout_Type shows minimal differentiation with all four modalities clustered within a 25-calorie range around the mean. These relationships reveal that the dataset structure prioritizes duration-based modeling of performance while incorporating modest influences from skill level and workout modality, creating a hierarchical system where time commitment drives the majority of caloric variation with experience and exercise type playing supporting roles.
