import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#from azr import *
from amazon_headphones_recommendation import *
from amazon_laptop_recommendation import *

app = Flask(__name__)

head_id_list = user_head_id
lap_id_list = user_lap_id

@app.route('/')
def home():
    return render_template('index.html',err='e')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    name = request.form['name']
    features = request.form['id']

    if name.lower() == 'headphones':
        try:
            prediction_head = recommend_it_head(preds_df_head, items_df_head, df1_head, 5,features)
            output = prediction_head
            return render_template('index.html',user_id=features, tables=[output.to_html(classes='data')], titles=prediction_head.columns.values)
        except KeyError:
            error='ID Not Found'
            return render_template('index.html',user_id=features,error=error)
    elif name.lower() == 'laptops':
        try:
            prediction_lap = recommend_it_lap(preds_df_lap, items_df_lap, df1_lap, 5,features)
            output = prediction_lap        
            return render_template('index.html',user_id=features, tables=[output.to_html(classes='data')], titles=prediction_lap.columns.values,lap=lap_id_list)
        except KeyError:
            error='ID Not Found'
            return render_template('index.html',user_id=features,error=error)
    else:
        error='No Products'
        return render_template('index.html',user_id=features,error=error)

        
if __name__ == "__main__":
    app.run(debug=True)