from flask import Flask, render_template, request

import pickle

application = Flask(__name__)
app = application


standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
ridge_model = pickle.load(open('models/ridgeCV.pkl', 'rb'))

@app.route('/')
def home():
    return "<h1> Welcome Pavan <h1>"

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        try:
            new_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data)
            return render_template('home.html', result=result[0])
        except Exception as e:
            return str(e)

    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")