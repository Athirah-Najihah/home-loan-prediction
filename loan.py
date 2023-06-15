# Final Assessment BCI3333
# Athirah Najihah binti Zulkifili
# loan_sanction_train.csv (https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval?select=loan_sanction_train.csv)

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Title and description
st.title('Home Loan Approval Prediction')
st.write('A dataset containing information about home loan applicants and their loan approval status.')

st.write("""
        Features: 
        - Married: Categorical variable indicating the marital status of the applicant
            - Options: No, Yes
        - Dependents: Number of dependents the applicant has
            - Options: 0, 1, 2, 3+
        - ApplicantIncome: Income of the applicant
            - Range: 0 to 100,000
        - CoapplicantIncome: Income of the co-applicant
            - Range: 0 to 50,000
        - LoanAmount: Amount of loan requested
            - Range: 0 to 100,000
        - Loan_Amount_Term: Term of the loan in months
            - Range: 0 to 360
        - Credit_History: Binary variable indicating the credit history of the applicant (0 = No, 1 = Yes)
            - Options: 0, 1
        - Property_Area: Categorical variable indicating the area of the property
            - Options: Rural, Semiurban, Urban
        - Loan_Status: Binary variable indicating the loan approval status (0 = Not Approved, 1 = Approved)
        """)

# Load the dataset
df = pd.read_csv('loan_sanction_train.csv')

# Drop unnecessary columns
df = df[['Married', 'Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']]

# Sidebar - User input features
st.sidebar.header('User Input Features')

# Collect user input features
married = st.sidebar.selectbox('Married', ('No', 'Yes'))
dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
applicant_income = st.sidebar.slider('Applicant Income', 0, 100000, 50000, 1000)
coapplicant_income = st.sidebar.slider('Coapplicant Income', 0, 50000, 10000, 1000)
loan_amount = st.sidebar.slider('Loan Amount', 0, 100000, 50000, 1000)
loan_amount_term = st.sidebar.slider('Loan Amount Term', 0, 360, 180, 1)
credit_history = st.sidebar.selectbox('Credit History', (0, 1))
property_area = st.sidebar.selectbox('Property Area', ('Rural', 'Semiurban', 'Urban'))

data = {
    'Married': married,
    'Dependents': dependents,
    'ApplicantIncome': str(applicant_income),
    'CoapplicantIncome': str(coapplicant_income),
    'LoanAmount': str(loan_amount),
    'Loan_Amount_Term': str(loan_amount_term),
    'Credit_History': credit_history,
    'Property_Area': property_area
}

input_df = pd.DataFrame(data, index=[0])

# Encode categorical variables
input_df['Married'] = input_df['Married'].map({'No': 0, 'Yes': 1})
input_df['Dependents'] = input_df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
input_df['Property_Area'] = input_df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})

# Load the saved model
load_clf = pickle.load(open('loan_svm.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)

# Display the input data graph
st.subheader('Input Data')
input_fig, input_ax = plt.subplots()
input_ax.bar(list(data.keys()), list(map(str, data.values())))
input_ax.set_ylabel('Values')
input_ax.set_title('Input Data')
plt.xticks(rotation=45)

# Display the input data graph
st.pyplot(input_fig)

# Display prediction and user input
st.subheader('Prediction')
status = {0: 'Not Approved', 1: 'Approved'}
st.write('The loan application is', '**' + status[prediction[0]] + '**')
