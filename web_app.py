from flask import Flask, render_template,request
import numpy as np
import os 
import joblib
import pandas as pd
from datetime import datetime

app=Flask(__name__)

result=""

def save_prediction_results(input_data, prediction_results):
    # Create a dictionary with all the data
    data = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **input_data,
        'Prediction': prediction_results.get('status', ''),
        'EMI': prediction_results.get('emi', ''),
        'PROI': prediction_results.get('proi', ''),
        'ELA': prediction_results.get('ela', '')
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Define file paths
    csv_path = 'Results/predictions.csv'
    excel_path = 'Results/predictions.xlsx'
    
    # Create Results directory if it doesn't exist
    os.makedirs('Results', exist_ok=True)
    
    # Save to CSV
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    
    # Save to Excel
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name='Predictions', header=False, index=False, startrow=writer.sheets['Predictions'].max_row)
    else:
        df.to_excel(excel_path, sheet_name='Predictions', index=False)


@app.route('/',methods=['GET'])
def welcome():
    return render_template("landing.html")

@app.route('/apply',methods=['GET'])
def apply():
    return render_template("index1.html")

@app.route('/signup',methods=['GET'])
def signup():
    return render_template("login.html")

@app.route('/login',methods=['GET'])
def login():
    return render_template("login.html")

@app.route('/about',methods=['GET'])
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        X = []
        input_data={}

        clsmodel = joblib.load('classification_model.joblib')

        rgrmodel = joblib.load('ridge_model_new.joblib')

        columns = ['CreditGrade','BorrowerAPR', 'BorrowerRate',
                   'LenderYield', 'ProsperScore', 'CreditScore', 'MonthlyLoanPayment',
                   'LP_CustomerPayments','LP_InterestandFees', 'LP_ServiceFees',
                   'LP_CustomerPrincipalPayments', 'LP_CollectionFees' , 
                   'LP_GrossPrincipalLoss', 'LoanOriginalAmount',
                   'StatedMonthlyIncome']

        for column in columns:
            value=request.form.get(column)
            input_data[column] = float(value)
            X.append(float(value))

        # print("X values = ", X)

        X = np.array(X)
        X = X.reshape(1, -1)
        test_arr = X

        yreg_pred = rgrmodel.predict(test_arr)
        ycls_pred = clsmodel.predict(test_arr)

        yreg_pred=np.array(yreg_pred)
        EMI = np.round(yreg_pred[:, 0],3)
        PROI = np.round(yreg_pred[:, 1],2)
        ELA = np.round(yreg_pred[:, 2],2)
        

        ELA=ELA/10

        # print("Cls Model Object: ", clsmodel)
        # print("Rgr Model Object: ", rgrmodel)
        prediction_results={}
        if EMI[0] < 0 or PROI[0] < 0 or ELA[0] < 0:
            predicted = "Defaulted"
            prediction_results = {
                'status': predicted,
                'emi': None,
                'proi': None,
                'ela': None
            }
            result = f"""
            <div class='result-item'><h3>Loan Status</h3>
            <div class='value' style='font-weight:700;font-size:1.1em;'>
            <span style='font-size:1.5em;vertical-align:middle;'>❌</span> {predicted}</div></div>
            """

        elif ycls_pred==0:
            predicted="Not Defaulted"
            prediction_results = {
                'status': predicted,
                'emi': float(str(EMI)[1:-1]),
                'proi': float(str(PROI)[1:-1]),
                'ela': float(str(ELA)[1:-1])
            }
            result = f"""
            <div class='result-item'><h3>Loan Status</h3><div class='value' style='font-weight:700;font-size:1.1em;'><span style='font-size:1.5em;vertical-align:middle;'>✅</span> {predicted}</div></div>
            <div class='result-item'><h3>Monthly EMI</h3><div class='value'>${float(str(EMI)[1:-1]):.2f}</div></div>
            <div class='result-item'><h3>PROI</h3><div class='value'>{float(str(PROI)[1:-1]):.2f}%</div></div>
            <div class='result-item'><h3>ELA</h3><div class='value'>{float(str(ELA)[1:-1]):.2f}</div></div>
            """
        else:
            predicted="Defaulted"
            prediction_results = {
                'status': predicted,
                'emi': None,
                'proi': None,
                'ela': None
            }
            result = f"""
            <div class='result-item'><h3>Loan Status</h3><div class='value' style='font-weight:700;font-size:1.1em;'><span style='font-size:1.5em;vertical-align:middle;'>❌</span> {predicted}</div></div>
            """
            
        save_prediction_results(input_data, prediction_results)
        
        
        return render_template('index1.html', result=result)
    return render_template('index1.html')

if __name__ == '__main__':
    app.run( debug=True)

