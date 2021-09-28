from flask import Flask, render_template, request
import pickle
import pandas as pd
#from sklearn.preprocessing import StandardScalar
#from flask_cors import CORS
app = Flask(__name__)
model = pickle.load(open('RFmodelForPrediction.sav', 'rb'))
scalar = pickle.load(open('RFstandardScalar.sav', 'rb'))
pca_model= pickle.load(open('RFpca_model.sav', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def predict():
    if request.method == 'POST':
        resp = request.form

        fixed_acidity = float(resp.get('fixed acidity'))
        volatile_acidity = float(request.form['volatile acidity'])
        citric_acid = float(request.form['citric acid'])
        residual_sugar = float(request.form['residual sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free sulfur dioxide'])
        total_sulfur_dioxide = float(request.form['total sulfur dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        l1 = [fixed_acidity , volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
              total_sulfur_dioxide, density, pH, sulphates, alcohol]
        temp_df = pd.DataFrame(l1)
        temp_df1 = temp_df.transpose()
        scaled_data = scalar.transform(temp_df1)
        principal_data = pca_model.transform(scaled_data)
        predict = model.predict(principal_data)
        if predict[0] == 0:
            result = 'Bad'
            print(result)
        else:
            result = 'Good'
            print(result)
        return render_template('index.html', wine_quality="WINE Quality is {}".format(result))

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)