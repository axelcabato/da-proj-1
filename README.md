# Gym Members Exercise Analysis

A data analysis project exploring gym member workout patterns, physical attributes, and fitness metrics. This project demonstrates the complete data analysis workflow from exploratory analysis through statistical testing to actionable business recommendations.

## Project Status

**Complete**

This is my first data analysis portfolio project, developed as I transition into the data science field. The project follows industry-standard practices and demonstrates a methodical approach to extracting insights from data.

| Phase | Status |
|-------|--------|
| Exploratory Data Analysis | Complete |
| Data Visualization | Complete |
| Feature Engineering | Complete |
| Data Standardization | Complete |
| Aggregated Statistics | Complete |
| Hypothesis Testing | Complete |
| Business Recommendations | Complete |

## Project Overview

This analysis examines a dataset of 973 gym members, investigating relationships between workout characteristics, physical attributes, and performance metrics. The project addresses questions such as:

- What factors most strongly predict caloric expenditure during workouts?
- How do performance metrics vary across experience levels and workout types?
- Are observed differences between demographic groups statistically significant?

### Key Finding

Session duration emerges as the dominant predictor of calories burned, with a correlation coefficient of 0.91, accounting for approximately 83% of the variance in caloric expenditure. Experience level shows systematic progression in energy output, while workout type demonstrates no statistically significant difference in calorie burn.

## Technical Approach

### Data Quality Assessment

The analysis begins with data profiling, including structural validation, missing value detection, and categorical integrity checks. I identified and documented that this dataset is synthetic, generated from published research averages. This recognition informed all subsequent analysis and conclusions.

### Exploratory Data Analysis

The EDA phase covers three areas:

**Univariate Analysis**: Distribution examination of numerical features including age, weight, calories burned, session duration, and body composition metrics.

**Categorical Analysis**: Frequency distributions across gender, workout type, experience level, and workout frequency. Confirmed balanced representation across categories.

**Bivariate Analysis**: Correlation analysis and relationship mapping between variables, including correlation matrices, grouped comparisons, and scatter plot visualizations.

### Feature Engineering

Created four derived features to support deeper analysis:

| Feature | Description | Purpose |
|---------|-------------|---------|
| `Calorie_Efficiency` | Calories burned per hour | Normalize performance across session lengths |
| `BMI_Category` | WHO standard classifications | Enable health-risk segmentation |
| `Age_Group` | Life-stage groupings | Facilitate demographic analysis |
| `Intensity_Score` | Heart rate capacity utilization | Quantify workout effort |

### Statistical Methods

- Descriptive statistics and distribution analysis
- Correlation analysis (Pearson)
- Data standardization (StandardScaler)
- Independent samples t-test
- One-way ANOVA

## Repository Structure

```
├── 1stDAProject.ipynb    # Main analysis notebook
├── 1stDAProject.py       # Python script version (Jupytext sync)
├── data/
│   └── gym_members_exercise_tracking.csv
├── .gitignore
├── .gitattributes
└── README.md
```

## Tools and Technologies

**Languages**: Python 3.13

**Data Manipulation**: pandas, NumPy

**Visualization**: Matplotlib, Seaborn

**Statistical Analysis**: SciPy, scikit-learn

**Development Environment**: Jupyter Notebook, VS Code

## Data Source

Dataset sourced from Kaggle: [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

The dataset is synthetic, generated using averages from publicly available fitness studies and industry reports. All conclusions acknowledge this limitation and note that findings would require validation against real-world data before practical application.

## What This Project Demonstrates

- Systematic approach to data exploration with documented methodology
- Recognition of data limitations and appropriate caveats on conclusions
- Application of Python data science tools including pandas, seaborn, matplotlib, and scikit-learn
- Clear documentation of findings with visualizations designed for both technical and general audiences
- Version control with Git and organized project structure

## About the Author

I am building skills in data analysis with a background in business administration and marketing. This project represents my first step into the data science field and my approach to learning: methodical, well-documented, and focused on practical skills.

I welcome feedback from experienced data professionals.

## Connect

**LinkedIn**: [linkedin.com/in/axelcabato](https://linkedin.com/in/axelcabato)

**Email**: contact@axelcabato.com

---

*Last updated: June 2026*