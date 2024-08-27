#!/usr/bin/env python
# coding: utf-8

# ### CSCT Masters Project 23Sep-UFCF9Y-60-M
# #### Student_Number:23044930
# #### Title: Evaluation of Model Bias and Fairness in Machine Learning Models: A study based on Bias Mitigation

# ### Content
#  #### 1. Initialisation
#  #### 2. Preprocessing of Data(Load Data).
#  #### 3. Initial Bias Check and Quantification of Bias Percentage.
#  #### 4. Applying Logistic Regression, Decision Tree, Random Forest and Gradient Boosting Models for Bias quantification.
#  #### 5. StreamLit App Interface.
# 

# ### 1. Initialisation

# In[ ]:


# Import necessary libraries for the project
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds


# In[ ]:


# Set a random seed for reproducibility to ensure consistent results across runs
np.random.seed(42)

# Dynamic patch to address an issue with the Fairlearn library
import fairlearn.reductions._exponentiated_gradient.exponentiated_gradient as eg
if hasattr(np, 'PINF'):
    np.PINF = np.inf
eg.np.PINF = np.inf


# In[ ]:


# Sidebar explanation of fairness constraints
st.sidebar.subheader("Fairness Constraints Explanations")
st.sidebar.markdown("""
- **Demographic Parity**: Ensures that the selection rates (e.g., positive predictions) are similar across different groups. It may lead to reduced accuracy because it doesn't account for true outcomes.
  
- **Equalized Odds**: Ensures that true positive rates and false positive rates are similar across groups. It can improve fairness but might require more complex adjustments to the model.

**Recommendations**:
- Choose **Logistic Regression** if interpretability is key and the data is not too complex.
- Use **Random Forest** or **Gradient Boosting** if you prioritize accuracy and can handle longer training times.
- **Demographic Parity** is recommended if ensuring equal opportunity across groups is more important than overall accuracy.
- **Equalized Odds** is better if you want to ensure fairness in the model's error rates across groups.
- Experiment with different models and fairness constraints to see which combination best meets your needs.
""")

# Display an 'About' section to introduce the concept of bias in machine learning models
def about_section():
    """Displays an 'About' section at the top of the app to introduce the concept of bias."""
    st.markdown("# What is Bias?")
    st.markdown(
        """
        Bias in machine learning refers to the systematic error that leads a model to make incorrect assumptions 
        about data. This can result from the way data is collected, processed, or how a model is trained. 
        Our tool provides a quantified figure to measure bias in models, helping you understand how fairly 
        your model treats different groups defined by sensitive features such as gender, race, or age.
        """
    )


# ### 2. Preprocessing of Data (Load Data).

# In[ ]:


# Load and preprocess the data
def load_and_preprocess_data(uploaded_file):
    """Loads the uploaded CSV file and handles any errors during the process.
    
    Args:
        uploaded_file: The uploaded file object.
    
    Returns:
        data: The loaded data as a pandas DataFrame, or None if an error occurs.
    """
    try:
        # Attempt to load the data from the uploaded file
        data = pd.read_csv(uploaded_file)
        
        # Preprocess the dataset
        # Replace placeholder '?' with NaN and drop rows with missing values
        data = data.replace('?', pd.NA).dropna()
        
    except pd.errors.EmptyDataError:
        # Display an error if the file is empty or incorrectly formatted
        st.error("The uploaded file is empty or malformed.")
        return None
    
    return data


# The columns are encoded to numerical values for the purpose of model training and testing.

# In[ ]:


# Encode categorical columns and retain the mapping for the sensitive feature
def encode_columns(data, sensitive_feature_column):
    """Encodes categorical columns in the dataset and keeps the mapping for the sensitive feature.
    
    Args:
        data: The dataset as a pandas DataFrame.
        sensitive_feature_column: The column name of the sensitive feature.

    Returns:
        data_encoded: The encoded dataset.
        sensitive_feature_mapping: A dictionary mapping encoded values to original categories for the sensitive feature.
    """
    # Create a copy of the dataset to avoid modifying the original data
    data_encoded = data.copy()
    
    # Encode all categorical features except the sensitive feature
    for column in data_encoded.columns:
        if data_encoded[column].dtype == object and column != sensitive_feature_column:
            data_encoded[column] = LabelEncoder().fit_transform(data_encoded[column])
    
    # Encode the sensitive feature and retain the mapping of the original categories
    le = LabelEncoder()
    data_encoded[sensitive_feature_column] = le.fit_transform(data_encoded[sensitive_feature_column])
    sensitive_feature_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    
    return data_encoded, sensitive_feature_mapping


# ### Initial Bias Check and Quantification of Bias Percentage.

# In[ ]:


def check_bias_logistic_regression(data, target_column, sensitive_feature_column, sensitive_feature_mapping):
    """Performs an initial bias check using Logistic Regression.
    
    This function trains a Logistic Regression model on the dataset and evaluates bias using 
    fairness metrics like selection rate, true positive rate, and disparate impact. It also
    plots a bar graph to show which group is favored based on the selected metric.
    
    Args:
        data: The dataset as a pandas DataFrame.
        target_column: The column name of the target variable.
        sensitive_feature_column: The column name of the sensitive feature.
        sensitive_feature_mapping: The mapping of encoded values to original categories for the sensitive feature.
    
    Returns:
        A dictionary containing the results of the bias analysis, including disparate impact and bias percentage.
    """
    # Split data into training and testing sets, with stratification based on the sensitive feature
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        data.drop(target_column, axis=1), data[target_column], data[sensitive_feature_column], 
        test_size=0.25, random_state=42
    )

    # Initialize and train the Logistic Regression model on the training set
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Define fairness metrics to evaluate the model's performance across different groups
    metrics = {
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
        'true_negative_rate': true_negative_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }

    # Create a MetricFrame to evaluate the model's fairness across sensitive groups
    metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)
    
    # Get the selection rate by group from the metric frame
    sr_by_group = metric_frame.by_group['selection_rate']

    # Map the numeric index back to the original category labels
    sr_by_group.index = sr_by_group.index.map(lambda x: sensitive_feature_mapping[x])

    # Calculate disparate impact and bias percentage for the sensitive feature
    disparate_impact = sr_by_group.min() / sr_by_group.max()
    bias_percentage = (1 - disparate_impact) * 100

    # Plotting the selection rate across different sensitive groups
    fig, ax = plt.subplots()
    sr_by_group.plot(kind='bar', color='skyblue', ax=ax)
    plt.title('Selection Rate by Sensitive Group')
    plt.ylabel('Selection Rate')
    plt.xlabel('Sensitive Group')
    st.pyplot(fig)

    # Return the model, metric frame, and calculated bias metrics for further analysis
    return {
        "Disparate Impact": disparate_impact,
        "Bias Percentage": bias_percentage,
        "Metric Frame": metric_frame,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "Sensitive Train": sensitive_train, "Sensitive Test": sensitive_test
    }


# ### Applying Logistic Regression, Decision Tree, Random Forest and Gradient Boosting for Bias quantification.

# In[ ]:


# Train multiple models and apply fairness constraints
def apply_fairness_constraints(data, target_column, sensitive_feature_column, constraint_option):
    """Trains multiple machine learning models and applies the selected fairness constraint.
    
    This function trains four different models, applies fairness constraints, and evaluates their 
    performance both before and after bias mitigation.
    
    Args:
        data: The dataset as a pandas DataFrame.
        target_column: The column name of the target variable.
        sensitive_feature_column: The column name of the sensitive feature.
        constraint_option: The fairness constraint to be applied (Demographic Parity or Equalized Odds).
    """
    # Define features (X) and the target variable (y) for the models
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    sensitive_feature = data[sensitive_feature_column]

    # Define the machine learning models to be trained
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Train and evaluate each model separately
    for model_name, model in models.items():
        st.markdown(f"## {model_name}")
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X, y, sensitive_feature, test_size=0.25, random_state=42)

        # Train the model on the training set
        model.fit(X_train, y_train)
        
        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Calculate initial fairness metrics before bias mitigation
        metrics = {
            'selection_rate': selection_rate,
            'true_positive_rate': true_positive_rate,
            'true_negative_rate': true_negative_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }
        metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)
        sr_by_group = metric_frame.by_group['selection_rate']
        disparate_impact = sr_by_group.min() / sr_by_group.max()
        bias_percentage = (1 - disparate_impact) * 100

        # Display the original accuracy and fairness metrics
        st.write(f"**Original Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
        st.write(f"**Original Bias Percentage**: {bias_percentage:.4f}%")
        st.write(f"**Original Disparate Impact**: {disparate_impact:.4f}")

        # Apply the selected fairness constraint to mitigate bias
        if constraint_option == "Demographic Parity":
            constraint = DemographicParity()
        else:
            constraint = EqualizedOdds()

        # Use ExponentiatedGradient to apply the fairness constraint and retrain the model
        mitigator = ExponentiatedGradient(model, constraint)
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
        y_pred_mitigated = mitigator.predict(X_test)

        # Calculate metrics after bias mitigation
        accuracy_mitigated = accuracy_score(y_test, y_pred_mitigated)
        precision = precision_score(y_test, y_pred_mitigated)
        recall = recall_score(y_test, y_pred_mitigated)
        f1 = f1_score(y_test, y_pred_mitigated)

        # Evaluate fairness metrics after bias mitigation
        metric_frame_mitigated = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=sensitive_test)
        sr_by_group_mitigated = metric_frame_mitigated.by_group['selection_rate']
        disparate_impact_mitigated = sr_by_group_mitigated.min() / sr_by_group_mitigated.max()
        bias_percentage_mitigated = (1 - disparate_impact_mitigated) * 100

        # Display the mitigated accuracy and fairness metrics
        st.write(f"**Mitigated Accuracy**: {accuracy_mitigated:.4f}")
        st.write(f"**Mitigated Bias Percentage**: {bias_percentage_mitigated:.4f}%")
        st.write(f"**Mitigated Disparate Impact**: {disparate_impact_mitigated:.4f}")
        st.write(f"**Precision**: {precision:.4f}")
        st.write(f"**Recall**: {recall:.4f}")
        st.write(f"**F1 Score**: {f1:.4f}")
        st.markdown("---")  # Divider between models


# ### StreamLit App Interface.

# In[ ]:


# Main Streamlit app function to run the entire application
def main():
    # Display the "About" section at the top
    about_section()
    
    # Title of the Streamlit app
    st.title("Flexible Fairness Analysis Tool")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and preprocess the uploaded data
        data = load_and_preprocess_data(uploaded_file)
        if data is not None:
            # Display an overview of the dataset
            st.write("### Dataset Overview")
            st.dataframe(data.head())

            # Allow the user to select the target and sensitive feature columns
            st.write("## Select Columns for Analysis")
            target_column = st.selectbox("Select the Target Column (e.g., income, credit_risk)", options=data.columns)
            sensitive_feature_column = st.selectbox("Select the Sensitive Feature Column (e.g., gender, race)", options=data.columns)

            # Button to check bias using Logistic Regression
            check_bias = st.button("Check Bias")
            if check_bias or st.session_state.get('bias_checked', False):
                if not st.session_state.get('bias_checked', False):
                    # Encode the data and perform initial bias analysis
                    data_encoded, sensitive_feature_mapping = encode_columns(data, sensitive_feature_column)
                    bias_results = check_bias_logistic_regression(data_encoded, target_column, sensitive_feature_column, sensitive_feature_mapping)
                    
                    # Display the results of the bias analysis
                    st.write("## Bias Analysis Results (Logistic Regression)")
                    st.write(f"Disparate Impact: {bias_results['Disparate Impact']:.2f}")
                    st.write(f"Bias Percentage: {bias_results['Bias Percentage']:.2f}%")
                    
                    # Store the state after bias checking
                    st.session_state['bias_checked'] = True
                    st.session_state['data_encoded'] = data_encoded
                    st.session_state['target_column'] = target_column
                    st.session_state['sensitive_feature_column'] = sensitive_feature_column
                    st.session_state['sensitive_feature_mapping'] = sensitive_feature_mapping

                # Allow the user to choose a fairness constraint for bias mitigation
                st.write("## Select Fairness Constraint for Bias Mitigation")
                constraint_option = st.selectbox("Select Fairness Constraint", ["Demographic Parity", "Equalized Odds"])

                # Button to apply the selected fairness constraint to all models
                apply_fairness = st.button("Apply Fairness Constraint to All Models")
                if apply_fairness:
                    # Retrieve the encoded data and column selections from the session state
                    data_encoded = st.session_state['data_encoded']
                    target_column = st.session_state['target_column']
                    sensitive_feature_column = st.session_state['sensitive_feature_column']
                    sensitive_feature_mapping = st.session_state['sensitive_feature_mapping']
                    
                    # Apply fairness constraints and display the results
                    apply_fairness_constraints(data_encoded, target_column, sensitive_feature_column, constraint_option)

# Run the app
if __name__ == "__main__":
    main()


# 
