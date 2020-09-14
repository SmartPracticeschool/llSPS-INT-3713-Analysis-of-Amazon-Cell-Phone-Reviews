import numpy as np
from flask import Flask, request, render_template
from joblib import load
import joblib
from tensorflow.keras.models import load_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)

model=load('nlp.h5')




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/page2',methods=['POST'])
def page2():
    '''
    For rendering results on HTML GUI
    '''
    da = request.form['Review']
    print(da)
    loaded=joblib.load('CountVectorizer')
    da=da.split("delimiter")
    result=model.predict(loaded.transform(da))
    prediction=result>=0.5
    print(prediction)
    if prediction[0] >= 0.5:
        output="Positive"
    elif prediction[0] <=0.5:
        output="Negative"

    return render_template('index.html', prediction_text='{}'.format(output))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():

    #For direct API calls trought request

    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
