from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import catboost
# from flask_cors import CORS
import os
import numpy as np
import pickle
import pandas as pd

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


def feature_Engineering_rf(data):
    """ Feature  Engineering for data to be predicted by random forest model """
    convert_to_categorical = ['Occupation','Product_Category_1','Product_Category_2','Marital_Status','Age','Gender','Stay_In_Current_City_Years','City_Category']
    for col in convert_to_categorical:
        data[col] = data[col].astype(str)
    data = data.drop('Product_Category_3',axis =1)
    data = pd.get_dummies(data,prefix_sep = '_',
                            columns = [col for col in data.columns if col not in ['User_ID','Product_ID']])
    data = data.drop(['User_ID','Product_ID'],axis =1)
    
    return data
def feature_Engineering_catBoost(data):
    """Feature engineering for data to be predicted using the catBoost Regressor model"""
    data = data.drop(['Product_Category_3','User_ID'], axis =1)
    return data
def feature_Engineering_lgbm(data):
    lab_enc = LabelEncoder()
    data = data.drop(['User_ID','Product_ID','Product_Category_3'],axis = 1)
    for col in data.columns:
        data[col] = lab_enc.fit_transform(data[col])
    return data
def ValuePredictor_rf(data):
    # to_predict = np.array(to_predict_list).reshape(1,12)
    rf_model = pickle.load(open("rF.pkl","rb"))
    print(rf_model)
    res = rf_model.predict(data)
    return res[0]
def ValuePredictor_catBoost(data):
    cat_model = pickle.load(open("catBoost.pkl","rb"))
    res = cat_model.predict(data)
    return res[0]
def ValuePredictor_lgbm(data):
    lgbm_model =  pickle.load(open("lgb_enc.pkl","rb"))
    res = lgbm_model.predict(data)
    return res[0]

@app.route('/result',methods = ['POST'])
def result():  
    if request.method == 'POST':
        req_data = request.form.to_dict()
        
        data_df = pd.DataFrame(req_data,columns=['User_ID','Product_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years',
        'Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3'],index=['input']) 
        
        data = feature_Engineering_catBoost(data_df)
        print("length of col", len(data.columns))
        res = ValuePredictor_catBoost(data)
        return render_template("index.html",prediction=res)    

if __name__ == "__main__":
    app.run() 
