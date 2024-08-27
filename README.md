"# 23044930_CSCT-Masters-Project-23Sep-UFCF9Y-60-M" 

Download the files attached in the GitHub and proceed for analysis
Adult Income Dataset: Target Variable: Income; Sensitive feature: Sex
Bank Marketing Dataset: Target Variable: Ter_Deposit; Sensitive Feature: marital status
Ps: You can choose any sensitive feature but target variable works well only with Binary values as per the development thus far.


1. To start the Streamlit app, run:
   ```bash
   streamlit run Application.py --server.enableXsrfProtection false
   ```
   ### Required Python Libraries
   The following Python libraries are required for this project:
   - Streamlit
   - pandas
   - numpy
   - scikit-learn
   - fairlearn

2. Open the provided URL in your web browser to access the app.
3. Upload your dataset, select the necessary columns, and start analyzing bias in your models.

About: 
AI Fairness Analysis Tool
A web-based application for detecting and mitigating bias in machine learning models, 
focusing on Demographic Parity and Equalized Odds using various algorithms like Logistic Regression, 
Decision Trees, Random Forests, and Gradient Boosting.

Installation
Prerequisites
Python 3.8 or higher
Git

Contact
Created by Samadhika Ranaweera as a part of the masters thesis  - feel free to contact me! (sameeduc@gamil.com)

Acknowledgments
Streamlit - The framework used to build the app.
Fairlearn - The library used for fairness metrics and mitigation.
Special thanks to Doctor Jan Van Lent for guidance and support.