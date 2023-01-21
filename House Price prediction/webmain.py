import pandas as pd
from flask import Flask, render_template,request
import pickle



app= Flask(__name__)
data=pd.read_csv('cleandata.csv')
#pipe=pickle.load(open("RidgeModel.pkl",'rb'))


@app.route('/')
def index():
    locations =sorted(data['locality'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    print(location,bhk )
    input=pd.DataFrame([location,bhk],columns=['location','bhk'])
    prediction=pipe.predict(input)[0]
    return ""

if __name__=="__main__":
    app.run(debug=True,port=5001)