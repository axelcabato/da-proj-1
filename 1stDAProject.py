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
# *(all language subject to change)*
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
# The categorical features of `Gender` and `Workout_Type` contain a small and consistent set of unique values:
# - For `Gender`, it is a binary categorical value with only two disctinct classes ("Male" and "Female"). The absence of additional unique values, such as inconsistent spellings, abbreviations, or missing value placeholders, confirms the high degree of data consistency for this feature.
# - Similarly, `Workout_Type` also has a small amount of disctinct and consistently labeled classes: "Cardio", "Strength", "HIIT" and "Yoga". This categorical integrity ensures that the variable is ready for direct use in analysis or for a simple transformation into a quantitative format, such as one-hot encoding, without requiring a separate data cleaning stage.
#
# The counts across all combinations (`Gender` × `Workout_Type`) range from 106 to 135 also, indicating a relatively balanced representation across groups. A favorable characteristic for subsequent comparative analyses.

# %% [markdown]
# ---

# %% [markdown]
# #### Recognizing the Data Source & Context

# %% [markdown]
# While clean in structure, the dataset contains several potential biases, limitations, and quirks that a data analyst must consider. The primary bias is that the dataset was simulated and generated using averages from publicly available studies and industry reports. This means the data may under- or over-represent certain behaviors or characteristics.
# - For instance, the randomization of `Experience_Level` and `Workout_Frequency` might not perfectly reflect the actual distribution of gym members, where, for example, a large number might be beginners who work out less frequently. This synthetic nature is *the most significant limitation*, as it lacks the unpredictable and messy nuances of real human behavior.
#
# **Any insights or models derived from this dataset would need to be validated with *actual, real-world data* before being applied to a genuine scenario.**
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
# - Specifically, I will convert the `Weight (kg)` and `Height (m)` variables from their current metric system to their imperial counterparts. This will be done to ensure the data is standardized for any subsequent statistical analysis and for enhanced data visualization tailored to our target audience.

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
# I use the `.describe()` method to validate the newly-engineered `Weight (lbs)` and `Height (ft)` features, confirming that the new columns have a reasonable range of values and are correctly populated. This ensures the integrity of our dataset for subsequent analysis.
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
# This type of analysis will allow me to understand their individual distributions, central tendencies, and potential outliers. To accomplish this, I will generate histograms for each of the key numerical columns with a KDE (Kernal Density Estimate) curve to overlay each to show the smoothed distribution shape.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style for all plots in this cell
sns.set_style("whitegrid")


###  Data Preparation  ###

# Create a copy of the DataFrame to avoid modifying original
df_copy = df.copy()

# Define the numerical features to visualize
numerical_features = [
    "Age", "Weight (lb)", "Height (ft)", "Calories_Burned",
    "Session_Duration (hours)", "Fat_Percentage", "BMI", "Avg_BPM", 
    "Resting_BPM", "Workout_Frequency (days/week)"
]


###  Figure Setup  ###

# Create a 2x5 grid (10 subplots) to accommodate all features
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))

# Flatten the 2D array of axes into 1D for easier iteration
axes = axes.flatten()


###  Generate Historigrams  ###

# Loop through each feature and create its histogram
for i, feature in enumerate(numerical_features):
    ax = axes[i]
    
    # Create histogram with KDE overlay
    sns.histplot(
        data=df_copy,
        x=feature,
        kde=True,           # Add smoothed distribution curve
        ax=ax,
        color="skyblue"
    )
    
    # Set labels and title for each subplot
    ax.set_title(f"Distribution of {feature}", fontsize=14)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

# Hide any unused subplots (if features < grid spaces)
for j in range(len(numerical_features), len(axes)):
    axes[j].axis("off")


###  Formatting  ###

# Add overarching title for the entire figure
fig.suptitle(
    "Univariate Analysis: Histograms of Key Numerical Features",
    fontsize=20,
    y=1.02
)

# Prevent label overlap between subplots
plt.tight_layout()

# Render figure in notebook
plt.show()

# %% [markdown]
# ##### Key Insights:

# %% [markdown]
# 1. `Age`
#     - The distribution spans from 18 to 59 years with a mean of approximately 39 years, showing that the simulated sample was designed to represent individuals across the full active adult age spectrum. The relatively even distribution across this range suggests that the data generation process did not intentionally concentrate observations at any particular age, resulting in a sample that includes young adults, middle-aged individuals, and older adults in roughly similar proportions.
# 2. `Workout_Frequency (days/week)`
#     - The distribution ranges from two to five days per week, with the values appearing distributed across these four levels, and notably excludes both very low frequency exercisers (one day per week) and daily exercisers (six to seven days per week). This bounded range reflects a design choice in the data generation process to focus the simulation on individuals who maintain moderate, consistent exercise schedules, thereby creating a sample that represents sustainable commitment patterns rather than the full spectrum of possible attendance behaviors.
# 3. `Calories_Burned`
#     - The values exhibit substantial variation, ranging from approximately 300 to 1,700 calories per session, yet this variation shows an exceptionally strong positive correlation of 0.91 with `Session_Duration`, confirming that the data generation algorithm primarily tied caloric expenditure to workout length. While the simulation incorporated some additional variance beyond pure time-based calculation (likely representing programmed influences of workout intensity, type, and individual metabolic factors), approximately 83 percent of the calorie variation can be explained by session duration alone, with the remaining 17 percent reflecting other parameters built into the simulation model.
# 4. `BMI` & `Fat_Percentage`
#     - The values for each exhibit a relatively balanced spread from 10 percent to 35 percent around a mean of approximately 25 percent, while the BMI distribution displays a modest positive skew with most values concentrated in the 20 to 30 range and a tail extending toward higher values reaching approximately 50. This difference in distributional shapes suggests that the data generation process applied different randomization or constraint parameters to these two body composition metrics, with `Fat_Percentage` following a more symmetrical generation pattern while BMI incorporated a right-skewed distribution that produces occasional higher values within the simulated sample.

# %% [markdown]
# In conclusion, this simulated dataset models a prototypical individual centered around 39 years of age who completes workout sessions averaging 1.26 hours in duration and expends approximately 905 calories per session. The simulated population exhibits cardiovascular metrics consistent with moderate fitness levels, with Resting BPM values tightly clustered around 62 beats per minute. The dataset incorporates substantial variation in body composition, with `Weight (lb)` values spanning from 88 pounds to 285.8 pounds, reflecting the simulation's design to represent individuals across a wide spectrum of body sizes. Notably, the strong positive correlation of 0.91 between `Session_Duration` and `Calories_Burned` demonstrates that the data generation algorithm primarily modeled energy expenditure as a function of workout length, with session duration explaining approximately 83 percent of caloric variation while the remaining variance reflects programmed influences of intensity, workout type, and simulated individual differences.

# %% [markdown]
# ##### Categorical Analysis

# %% [markdown]
# Next, I move onto Categorical Analysis. I will utilize count plots (bar charts) to visualize the frequency distribution of the categorical variables. By understanding these distributions, it will allow me to understand how different groups behave as well as identify class balance and potential sampling biases present in the dataset.

# %%
# Set the visual style for all plots in this cell
sns.set_style("whitegrid")


# HELPER FUNCTION
def add_value_labels_inside(ax):
    """
    Adds count labels inside each bar of a bar plot.
    
    Placing labels inside bars (rather than above) keeps the visualization
    compact and makes values immediately readable without eye movement.
    
    Parameters:
        ax: matplotlib Axes object containing the bar plot
    """
    for patch in ax.patches:
        height = patch.get_height()
        
        # Position text at the center of each bar
        x = patch.get_x() + patch.get_width() / 2  # Horizontal center
        y = height / 2                              # Vertical center
        
        ax.text(
            x, y,
            f'{int(height)}',    # Display count as whole number
            ha='center',         # Horizontal alignment
            va='center',         # Vertical alignment
            fontsize=11,
            fontweight='bold',
            color='white'        # White text for contrast on colored bars
        )


###  Figure Setup  ###

# Create a 2x2 grid for four categorical variables
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# Flatten for easier indexing
axes = axes.flatten()


## Plot 1 (Top-Left): GENDER DISTRIBUTION
# Purpose: Verify balance between male and female gym members

sns.countplot(
    data=df,
    x='Gender',
    hue='Gender',
    ax=axes[0],
    palette='viridis',
    legend=False           # Legend redundant when x-axis shows categories
)

axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Gender', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)

add_value_labels_inside(axes[0])

## Plot 2 (Top-Right): WORKOUT TYPE DISTRIBUTION
# Purpose: Check representation across exercise modalities

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
axes[1].tick_params(axis='x', rotation=45)  # Angle labels to prevent overlap

add_value_labels_inside(axes[1])

## Plot 3 (Bottom-Left): EXPERIENCE LEVEL DISTRIBUTION
# Purpose: Understand skill level composition of gym membership
# Order: 1 → 2 → 3 (Beginner to Advanced) for intuitive left-to-right reading

sns.countplot(
    data=df,
    x='Experience_Level',
    hue='Experience_Level',
    ax=axes[2],
    palette='magma',
    legend=False,
    order=[1, 2, 3]     # Explicit ordering: Beginner → Intermediate → Advanced
)

axes[2].set_title('Experience Level Distribution', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Experience Level (1=Beginner, 2=Intermediate, 3=Advanced)', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)

add_value_labels_inside(axes[2])

## Plot 4 (Bottom-Right): WORKOUT FREQUENCY DISTRIBUTION
# Purpose: Examine how often members exercise per week
# Order: 2 → 5 days to show natural progression

sns.countplot(
    data=df,
    x='Workout_Frequency (days/week)',
    hue='Workout_Frequency (days/week)',
    ax=axes[3],
    palette='cividis',
    legend=False,
    order=[2, 3, 4, 5]     # Ascending order of workout frequency
)

axes[3].set_title('Workout Frequency Distribution', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Workout Frequency (days/week)', fontsize=12)
axes[3].set_ylabel('Count', fontsize=12)

add_value_labels_inside(axes[3])


###  Formatting  ###

# Add overarching title for the entire figure
fig.suptitle(
    'Categorical Analysis: Distribution of Key Features',
    fontsize=18,
    fontweight='bold',
    y=0.995
)

# Prevent label overlap between subplots
plt.tight_layout()

# Render figure in notebook
plt.show()

# %% [markdown]
# ##### Bivariate Analysis

# %% [markdown]
# Finally, I have decided to perform Bivariate Analysis to uncover potential correlations and dependencies within the dataset. By examining how one variable (like `Workout_Type`) relates to another (such as `Calories_Burned`), I can identify the mathematical relationships programmed into the data generation process. The following visualizations and statistical comparisons highlight key relationships embedded in the dataset structure.

# %%
# Set the visual style for all plots in this cell
sns.set_style("whitegrid")


####  Data Preparation  ###

# Define numerical columns for correlation analysis
numerical_cols = [
    'Age', 'Weight (lb)', 'Height (ft)', 'Max_BPM', 'Avg_BPM', 
    'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
    'Water_Intake (liters)', 'BMI', 'Fat_Percentage'
]

# Calculate correlation matrix for all numerical features
correlation_matrix = df[numerical_cols].corr()

# Aggregate calories burned by workout type for bar chart comparison
workout_type_summary = (
    df.groupby('Workout_Type')['Calories_Burned']
    .mean()
    .sort_values(ascending=False)  # Highest calorie-burning workout first
    .reset_index()
)


###  Figure Setup  ###

# Create 2x2 subplot grid with adequate spacing for labels
fig, axes = plt.subplots(2, 2, figsize=(16, 12))


# Plot 1 (Top-Left): CORRELATION HEATMAP
# Purpose: Show how strongly each variable correlates with key metrics
# Design choice: Transpose (.T) places target variables as rows for easier reading

# Select only the most analytically relevant variables for focused comparison
key_variables = ['Calories_Burned', 'Session_Duration (hours)', 'Avg_BPM', 'Fat_Percentage']

sns.heatmap(
    correlation_matrix[key_variables].T,  # Transpose: key vars as rows
    annot=True,                            # Display correlation coefficients
    fmt='.2f',                             # Two decimal places
    cmap='coolwarm',                       # Red = positive, Blue = negative correlation
    cbar=True,                             # Include color scale reference
    linewidths=0.5,                        # Grid line thickness
    linecolor='black',                     # Grid line color
    ax=axes[0, 0]
)

axes[0, 0].set_title('Correlation Matrix: Key Variable Relationships', 
                     fontsize=14, fontweight='bold')
axes[0, 0].tick_params(axis='y', rotation=0)   # Horizontal y-axis labels
axes[0, 0].tick_params(axis='x', rotation=45)  # Angled x-axis labels for readability

## Plot 2 (Top-Right): BAR CHART - CALORIES BY WORKOUT TYPE
# Purpose: Compare average calorie expenditure across workout modalities
# Design choice: Sorted descending to quickly identify highest-burning activities

sns.barplot(
    data=workout_type_summary,
    x='Workout_Type',
    y='Calories_Burned',
    hue='Workout_Type',      # Color-code by workout type
    ax=axes[0, 1],
    palette='viridis',       # Colorblind-friendly palette
    legend=False             # Redundant legend (x-axis already labeled)
)

axes[0, 1].set_title('Average Calories Burned by Workout Type', 
                     fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Workout Type', fontsize=12)
axes[0, 1].set_ylabel('Average Calories Burned', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)

# Add value labels inside each bar for precise reading
for patch in axes[0, 1].patches:
    height = patch.get_height()
    axes[0, 1].text(
        patch.get_x() + patch.get_width() / 2,  # Center horizontally
        height / 2,                              # Center vertically within bar
        f'{int(height)}',                        # Display as whole number
        ha='center', va='center',
        fontsize=11, fontweight='bold', color='white'
    )

## Plot 3 (Bottom-Left): BOX PLOT - CALORIES BY EXPERIENCE LEVEL
# Purpose: Show distribution and spread of calories burned across skill levels
# Design choice: Ordered 3→2→1 (Advanced first) to show progression visually

sns.boxplot(
    data=df,
    x='Experience_Level',
    y='Calories_Burned',
    order=[3, 2, 1],          # Display: Advanced → Intermediate → Beginner
    hue='Experience_Level',
    ax=axes[1, 0],
    palette='magma',
    legend=False
)

axes[1, 0].set_title(
    'Calories Burned Distribution by Experience Level', 
    fontsize=14, 
    fontweight='bold'
    )
axes[1, 0].set_xlabel(
    'Experience Level (3=Advanced, 2=Intermediate, 1=Beginner)', 
    fontsize=12
    )
axes[1, 0].set_ylabel('Calories Burned', fontsize=12)

## Plot 4 (Bottom-Right): SCATTER PLOT - BPM VS CALORIES
# Purpose: Explore relationship between heart rate intensity and calorie burn
# Design choice: Point size AND color both encode session duration to emphasize
#   its role as the dominant predictor of calories burned (r = 0.91)

scatter = sns.scatterplot(
    data=df,
    x='Avg_BPM',
    y='Calories_Burned',
    hue='Session_Duration (hours)',   # Color gradient by duration
    size='Session_Duration (hours)',  # Larger points = longer sessions
    sizes=(20, 200),                  # Size range (min, max)
    palette='viridis',
    ax=axes[1, 1],
    alpha=0.6                         # Transparency to show overlapping points
)

axes[1, 1].set_title(
    'Average BPM vs Calories Burned (by Session Duration)', 
    fontsize=14, 
    fontweight='bold'
)
axes[1, 1].set_xlabel('Average BPM', fontsize=12)
axes[1, 1].set_ylabel('Calories Burned', fontsize=12)
axes[1, 1].legend(
    title='Duration (hrs)', loc='lower right', fontsize=9)

# -----------------------------------------------------------------------------
# FINAL FORMATTING & OUTPUT
# -----------------------------------------------------------------------------

# Add overarching title for the entire figure
fig.suptitle('Key Bivariate Relationships', 
             fontsize=18, fontweight='bold', y=0.995)

# Prevent label overlap between subplots
plt.tight_layout()

# Render the figure in the notebook
plt.show()

# %% [markdown]
# ##### Key Insights
#
# 1. `Session_Duration` vs. `Calories_Burned`
#     - The correlation matrix reveals an exceptionally strong positive correlation of 0.91 between the two, confirming that the data generation algorithm primarily tied caloric expenditure to workout length. This relationship accounts for approximately 83 percent of the variance in calories burned, with the remaining 17 percent reflecting programmed influences of workout intensity, exercise type, and simulated individual metabolic differences.
#
# 2. `Workout_Type` vs. `Calories_Burned`
#     - The average calories burned across workout types shows HIIT at 926 calories, followed by Cardio at 885 calories, Strength at 911 calories, and Yoga at 903 calories. This relatively narrow range (41-calorie spread) indicates the simulation programmed some degree of differentiation in caloric output between workout modalities, with all four types clustering tightly around the overall mean of 905 calories.
#
# 3. `Experience_Level` vs. `Calories_Burned`
#     - The box plot distribution reveals that Advanced members (Level 3) have median calorie burns approximately 400 calories higher than Beginners (Level 1), with Intermediate members (Level 2) falling between these extremes. This systematic progression reflects a design choice to model experience level as a strong predictor of workout output, with higher experience correlating with both longer session durations and greater caloric expenditure.
#
# 4. `Average_BPM` vs. `Calories_Burned`
#     - The scatter plot demonstrates a positive relationship between the two, with color gradation showing that longer session durations correspond to both higher heart rates and greater caloric expenditure. This pattern indicates the simulation modeled cardiovascular intensity as interconnected with both workout duration and energy expenditure, creating a three-way relationship where time, heart rate, and calories increase together.

# %% [markdown]
# In conclusion, the bivariate relationships within this simulated dataset demonstrate that the data generation algorithm was programmed with `Session_Duration` as the dominant predictor of caloric expenditure, accounting for 83 percent of the variance through a strong positive correlation of 0.91. The simulation incorporated secondary variation through `Experience_Level`, which shows systematic progression in energy output from beginner to advanced classifications, while `Workout_Type` shows some degree of differentiation with all four modalities clustered within a 41-calorie range around the mean. These relationships reveal that the dataset structure prioritizes duration-based modeling of performance while incorporating modest influences from skill level and workout modality, creating a hierarchical system where time commitment drives the majority of caloric variation with experience and exercise type playing supporting roles.

# %% [markdown]
# ---

# %% [markdown]
# #### Data Transformation & Feature Engineering

# %% [markdown]
# Having completed the exploratory data analysis and visualization phases, I now move into data transformation to enhance the analytical value of this simulated dataset. This phase involves creating derived features that provide new perspectives on the data, standardizing measurements for fair comparison across different scales, and generating aggregated statistics that reveal patterns at the group level. These transformation steps prepare the dataset for deeper statistical analysis and demonstrate techniques commonly used in data analysis and engineering workflows to extract maximum insight from raw data.

# %% [markdown]
# ##### Creating Derived Features

# %% [markdown]
# Feature engineering involves creating new calculated columns from existing variables to provide additional analytical perspectives and answer specific business questions. By constructing metrics such as "Calorie Efficiency", BMI Classifications, and Composite Intensity Scores, I can transform raw measurements into meaningful indicators that support more nuanced analysis of workout performance and member characteristics.

# %%
# Calculate calorie efficiency (calories burned per hour)
df['Calorie_Efficiency'] = round(df['Calories_Burned'] / df['Session_Duration (hours)'], 2)

# Create BMI categories based on standard WHO classifications
df['BMI_Category'] = pd.cut(
    df['BMI'],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# Create Age Group categories
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 29, 44, 59],
    labels=['Young Adult (18-29)', 'Middle-Aged (30-44)', 'Senior (45-59)']
)

# Create Intensity Score (normalized combination of heart rate metrics)
df['Intensity_Score'] = round((
    (df['Avg_BPM'] - df['Resting_BPM']) / df['Max_BPM']
) * 100, 2)


# Verify new features were created successfully
df[['Calorie_Efficiency', 'BMI_Category', 'Age_Group', 'Intensity_Score']].head(10)

# %% [markdown]
# **Features Created:**
# - `Calorie_Efficiency`: Calories burned per hour of exercise
# - `BMI_Category`: Standard Body Mass Index (BMI) classifications (Underweight, Normal, Overweight, Obese). Based on the [World Health Organization (WHO)](https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html).
# - `Age_Group`: Life-stage groupings for demographic analysis
# - `Intensity_Score`: Percentage of heart rate capacity utilized during workout

# %% [markdown]
# ##### Normalize/Standardize Numerical Features

# %% [markdown]
# Variables in this dataset are measured on vastly different scales. `Age` ranges from 18 to 59, while `Calories_Burned` ranges from 300 to 1,700, making direct numerical comparisons problematic without standardization. Scaling transforms all features to a common range (typically, with a mean of zero and a standard deviation of one), ensuring that variables with larger numerical ranges do not artificially dominate analyses or visualizations that compare multiple metrics simultaneously.

# %%
from sklearn.preprocessing import StandardScaler

# Select numerical columns to standardize (excluding categorical and derived category features)
columns_to_scale: list = [
    'Age', 'Weight (lb)', 'Height (ft)', 'Max_BPM', 'Avg_BPM', 
    'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
    'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
    'Experience_Level', 'BMI', 'Calorie_Efficiency', 'Intensity_Score'
]

# Initialize the scaler
scaler = StandardScaler()

# Create new DataFrame with scaled values (preserving originals)
df_scaled = df.copy()
df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Verify standardization worked (mean ≈ 0, std ≈ 1)
df_scaled[columns_to_scale].describe().loc[['mean', 'std']].round(2)

# %% [markdown]
# The standardized features now share a common scale with means approximately equal to zero and standard deviations of one. The original DataFrame `df` remains unchanged for interpretability, while `df_scaled` is available for any analyses requiring normalized inputs.

# %% [markdown]
# ##### Create Aggregated Summary Statistics

# %% [markdown]
# While individual observations provide granular detail, strategic decision-making requires understanding patterns at the group and category levels through statistical aggregation. By calculating summary metrics across combinations of categorical variables, such as average performance by `Gender` and `Workout_Type`, or session characteristics by `Experience_Level`, I can identify trends and differences that inform targeted recommendations for distinct member segments.

# %%
# Summary 1: Performance metrics by Experience Level
experience_summary = df.groupby('Experience_Level').agg({
    'Calories_Burned': ['mean', 'median', 'std'],
    'Session_Duration (hours)': 'mean',
    'Calorie_Efficiency': 'mean',
    'Workout_Frequency (days/week)': 'mean'
}).round(2)

print("=== PERFORMANCE BY EXPERIENCE LEVEL ===")
print("(1 = Beginner, 2 = Intermediate, 3 = Advanced)\n")
display(experience_summary)

# Summary 2: Calories burned by Gender and Workout Type
gender_workout_summary = df.groupby(['Gender', 'Workout_Type']).agg({
    'Calories_Burned': 'mean',
    'Session_Duration (hours)': 'mean',
    'Intensity_Score': 'mean'
}).round(2)

print("\n=== PERFORMANCE BY GENDER AND WORKOUT TYPE ===\n")
display(gender_workout_summary)

# Summary 3: Age Group Analysis
age_group_summary = df.groupby('Age_Group', observed=True).agg({
    'Calories_Burned': 'mean',
    'BMI': 'mean',
    'Resting_BPM': 'mean',
    'Workout_Frequency (days/week)': 'mean'
}).round(2)

print("\n=== HEALTH METRICS BY AGE GROUP ===\n")
display(age_group_summary)

# %% [markdown]
# **Key Observations:**
#
# - `Experience_Level` shows clear progression: advanced members burn more calories, exercise longer, and work out more frequently than beginners.
# - `Gender` × `Workout_Type` reveals whether certain exercise modalities show performance differences between male and female members.
# - `Age_Group` comparisons highlight how fitness metrics and habits shift across life stages.
#
# These grouped summaries provide the foundation for targeted recommendations in the Business Insights section.

# %% [markdown]
# #### Statistical Interpretation & Hypothesis Testing

# %% [markdown]
# The exploratory analysis done previously revealed apparent differences in calorie expenditure across gender, workout type, and experience level. However, observed differences in sample data ***do not automatically*** indicate true population-level effects. They could result from random variation. Statistical hypothesis testing provides a rigorous framework to determine whether these patterns are statistically significant or likely attributable to chance.
#
# This section applies two common inferential tests:
# - **Independent Samples T-Test**: Evaluates whether male and female members differ significantly in calories burned
# - **One-Way ANOVA**: Evaluates whether significant differences exist in calories burned across the four workout types
#
# A significance threshold of α = 0.05 is used for all tests, meaning results with p-values below 0.05 are considered statistically significant.

# %%
from scipy import stats

# ============================================================
# TEST 1: Independent Samples T-Test (Gender vs Calories Burned)
# ============================================================
# H₀: No significant difference in calories burned between males and females
# H₁: Significant difference exists in calories burned between males and females

male_calories = df[df['Gender'] == 'Male']['Calories_Burned']
female_calories = df[df['Gender'] == 'Female']['Calories_Burned']

t_stat, t_pvalue = stats.ttest_ind(male_calories, female_calories)

print("=" * 60)
print("TEST 1: INDEPENDENT SAMPLES T-TEST")
print("Question: Do males and females burn significantly different calories?")
print("=" * 60)
print(f"Male mean:    {male_calories.mean():.2f} calories")
print(f"Female mean:  {female_calories.mean():.2f} calories")
print(
    f"Difference:   {abs(male_calories.mean() - female_calories.mean()):.2f} calories")
print(f"\nT-statistic:  {t_stat:.4f}")
print(f"P-value:      {t_pvalue:.4f}")
print(
    f"\nResult: {'SIGNIFICANT' if t_pvalue < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")

# ============================================================
# TEST 2: One-Way ANOVA (Workout Type vs Calories Burned)
# ============================================================
# H₀: No significant difference in calories burned across workout types
# H₁: At least one workout type differs significantly in calories burned

cardio = df[df['Workout_Type'] == 'Cardio']['Calories_Burned']
strength = df[df['Workout_Type'] == 'Strength']['Calories_Burned']
hiit = df[df['Workout_Type'] == 'HIIT']['Calories_Burned']
yoga = df[df['Workout_Type'] == 'Yoga']['Calories_Burned']

f_stat, anova_pvalue = stats.f_oneway(cardio, strength, hiit, yoga)

print("\n" + "=" * 60)
print("TEST 2: ONE-WAY ANOVA")
print("Question: Do workout types differ significantly in calories burned?")
print("=" * 60)
print(f"Cardio mean:   {cardio.mean():.2f} calories")
print(f"Strength mean: {strength.mean():.2f} calories")
print(f"HIIT mean:     {hiit.mean():.2f} calories")
print(f"Yoga mean:     {yoga.mean():.2f} calories")
print(f"\nF-statistic:   {f_stat:.4f}")
print(f"P-value:       {anova_pvalue:.4f}")
print(
    f"\nResult: {'SIGNIFICANT' if anova_pvalue < 0.05 else 'NOT SIGNIFICANT'} (α = 0.05)")

# %% [markdown]
# ##### Hypothesis Testing Results & Interpretation
#
# **Test 1 — Gender Comparison (T-Test):**
# The independent samples t-test evaluated whether calorie expenditure differs significantly between male and female gym members. The results indicate a statistically significant difference (t = 4.75, p < 0.001). On average, male members burned 944.46 calories per session compared to 862.25 for female members, a difference of 82.21 calories. Because p < 0.05, we reject the null hypothesis and conclude that gender is associated with a meaningful difference in calories burned.
#
# **Test 2 — Workout Type Comparison (ANOVA):**
# The one-way ANOVA evaluated whether mean calorie expenditure varies significantly across the four workout modalities. The results indicate no statistically significant difference (F = 0.95, p = 0.416). Although HIIT showed the highest mean (925.81 calories) and Cardio the lowest (884.51 calories), this 41-calorie spread is not large enough to rule out random chance. We fail to reject the null hypothesis, meaning workout type alone does not reliably predict calorie expenditure.
#
# *Important Caveats:*
# - Statistical significance *does not* imply practical significance. A difference can be "real" but too small to matter in application.
# - These tests assume the underlying data is approximately normally distributed. Given the synthetic nature of this dataset, this assumption is likely satisfied by design.
# - As stated previously, this dataset is synthetic (computer-generated), so these findings *should not* be generalized to real-world gym populations.
# - As noted in the EDA, the dominant predictor of calories burned is session duration (r = 0.91). These group comparisons do not account for session length, meaning observed differences may partly reflect variation in how long different groups exercise rather than inherent differences in calorie-burning efficiency.
#
# These statistical tests confirm which exploratory observations reflect genuine patterns versus random variation, providing an evidence-based foundation for the business recommendations that follow.

# %% [markdown]
# #### Business Insights & Recommendations
