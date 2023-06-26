import json
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from datetime import datetime

def days_between(d2):
    d1 = "2022-12-31"
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)
    
def return_predictions(time, city, date):
    df = pd.read_csv('data/EXL_EQ_2023_Dataset.csv')
    where = []
    for n, i in enumerate(list(df["Time Periods"])):
        if i[-5:] != time:
            where.append(n)
    df = df.drop(index=where)
    df = df[df['City'] == city]
    X = np.array(df["PM2.5"])
    cleanedList = [x for x in X if str(x) != 'nan']
    len(cleanedList)
    X = np.array(cleanedList)
    model = ARIMA(X, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.predict(start=X.shape[0], end=X.shape[0]+days_between(date)-1, dynamic=True)
    final_pred = predictions[days_between(date)-1]
    return final_pred

    
app = Flask(__name__)
@app.route('/health')
def index():
    return json.dumps({'Health': 'UP'})

@app.route('/api/return_predictions',methods=["GET"])
def return_prediction_get():
    request_data = request.json
    t = request_data['time']
    c = request_data['city']
    d = request_data['date']
    response =  return_predictions(t,c,d)
    print(response)
    response_data = {
        'status': 'success',
        'message': 'Request processed successfully',
        'response': json.dumps(response.tolist())
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)