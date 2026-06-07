# Diabetes Risk Prediction Pipeline & Web App


## Project Overview
This repository contains an end-to-end Data Mining and Machine Learning pipeline designed to predict a patient's risk of diabetes based on clinical and demographic data. This project was developed as a final academic project, fulfilling requirements for data exploration, preprocessing, algorithm tuning, and web deployment.

**Live Web Application:** [Click here to view the app](https://7ch6nx6o7khacbppsgvtvd.streamlit.app/)

**Live Web Application Demo:** https://youtu.be/iRzyBMTyBNo

**Presentation VIDEO:** https://youtu.be/QAGIwLhjhIA




## Repository Structure
* [`Colabproject.ipynb`](./Colabproject.ipynb): The Jupyter Notebook containing the full data science pipeline (EDA, Cleaning, Feature Engineering, Training, Evaluation).
* [`app.py`](./app.py): The Streamlit web application script.
* [`diabetes_dataset (1).csv`](./diabetes_dataset%20(1).csv): The original clinical dataset.
* [`diabetes_model_tuned.pkl`](./diabetes_model_tuned.pkl): The tuned Random Forest model saved via Joblib.
* [`scaler.pkl`](./scaler.pkl): The StandardScaler object used to normalize numerical inputs.
* [`label_encoders.pkl`](./label_encoders.pkl): The dictionary of LabelEncoders for categorical data.
* [`requirements.txt`](./requirements.txt): The dependencies required to run the web app.
  
  *Dataset Source*
The clinical data used to train this model is the **100,000 Diabetes Clinical Dataset**. 
* **Original Source:** [View and download the original dataset on Kaggle](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset)


## 1. Data Discovery & Cleaning
The dataset consists of 100,000 patient records. To prepare the data for modeling, several cleaning steps were required:
* **Missing Data:** Handled 35,000+ implicit missing values (labeled as "No Info" in smoking history).
* **Duplicates:** Identified and removed 14 duplicate patient records.
* **Anomalies:** Filtered out a statistically insignificant gender category ('Other', 18 rows) to prevent algorithmic skewing.
* **Encoding:** Applied `LabelEncoder` to transform text variables into machine-readable formats.

## 2. Exploratory Data Analysis & Feature Engineering
* **Univariate & Bivariate Analysis:** Explored variables like BMI, Age, and Blood Glucose, noting a distinct right-skew in age and a significant class imbalance (~91% non-diabetic).
* **Feature Engineering:** Created a brand new feature, `BMI_Category`, which classifies raw BMI numbers into standard medical categories (Underweight, Normal, Overweight, Obese).
* **Feature Selection:** Utilized an ANOVA F-value filter (`SelectKBest`) to identify the most statistically significant predictors, revealing Blood Glucose as the highest-impact feature.

## 3. Algorithm Training & Tuning
Three supervised learning algorithms were tested: **Logistic Regression, Decision Trees, and Random Forest**. 
* **Model Selection:** Random Forest was chosen for its superior ability to handle complex clinical relationships.
* **Hyperparameter Tuning:** Applied `GridSearchCV` to find the optimal parameters (`max_depth`, `n_estimators`, `min_samples_split`) using 3-fold cross-validation. This prevented overfitting and maximized the model's ability to generalize to unseen patients.

## 4. Validation & Evaluation
The model was validated using a 20% test holdout set. The primary focus was minimizing false negatives in a medical context. 
* **Precision:** Achieved > 0.94
* **Recall:** Achieved > 0.82
*(Both metrics successfully exceeded the 0.3 minimum requirement).*
