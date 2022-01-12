import os
import pandas as pd
import pickle
from flask import Flask, request, Response
from rossman.Rossman import Rossman

# carregando model
model = pickle.load(open('model/model_rossman.pkl', 'rb')) # relative path for Heroku

# iniciando API
app = Flask(__name__)

@app.route('/rossman/predict', methods=['POST'])
def rossman_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict): # se o json tem apenas uma linha
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instanciando a classe Rossman
        pipeline = Rossman()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000) # especificando ao Heroku qual porta utilizar
    app.run(host='0.0.0.0', port=port) 