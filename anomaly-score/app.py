# Importing libraries
from flask import Flask,request,render_template,jsonify,url_for,redirect
import pickle
import numpy as np
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Initializing the flask app and unpacking the model
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# Setting up the database for storage of anomaly score predictions
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
Migrate(app,db)

class PredictionModel(db.Model):
    # Building the storage model for anomaly score predicitons
    
    __tablename__= 'prediction'

    id = db.Column(db.Integer,primary_key=True)
    temperature = db.Column(db.Integer)
    humidity = db.Column(db.Integer)
    sound_volume = db.Column(db.Integer)
    anomaly_score = db.Column(db.Float)

    def __init__(self,temperature,humidity,sound_volume,anomaly_score):
        self.temperature = temperature
        self.humidity = humidity
        self.sound_volume = sound_volume
        self.anomaly_score = anomaly_score
    
@app.route('/api',methods=['POST'])
def prediction_api():
    # Building REST api for the model
    data = request.json['data']
    data = np.array(list(data.values())).reshape(1,-1)
    prediction = model.predict(data)
    return jsonify(prediction[0])

@app.route('/')
def home():
    # Setting up the home page (form page)
    return render_template('form.html')

@app.route('/prediction',methods=['POST','GET'])
def save_prediction():
    # Saving and displaying the prediction in form of a table
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        sound_volume = float(request.form['sound_volume'])
        values = np.array([[temperature,humidity,sound_volume]])
        anomaly_score = model.predict(values)
        save_data = PredictionModel(temperature,humidity,sound_volume,anomaly_score)
        db.session.add(save_data)
        db.session.commit()
        data = PredictionModel.query.all()
        return render_template('prediction.html',data=data)
    if request.method == 'GET':
        data = PredictionModel.query.all()
        return render_template('prediction.html',data=data)

# Running the app       
if __name__ == '__main__':
    app.run(debug=True)