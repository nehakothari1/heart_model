from flask import Flask, render_template, request
import pickle 
app = Flask(__name__)
model = pickle.load(open('heart.pkl','rb')) #read mode

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form["age"])
        sex = int(request.form["sex"]) 
        trestbps = int(request.form["trestbps"]) 
        chol =int(request.form["chol"]) 
        oldpeak = int(request.form["oldpeak"])  
        thalach = int(request.form["thalach"]) 
        fbs = int(request.form["fbs"])  
        exang = int(request.form["exang"]) 
        slope = int(request.form["slope"]) 
        cp = int(request.form["cp"])  
        thal = int(request.form["thal"]) 
        ca =int(request.form["ca"]) 
        restecg = int(request.form["restecg"])  
    input_cols =([[age, sex, cp, trestbps,  
                    chol, fbs, restecg, thalach,  
                    exang, oldpeak, slope, ca,  
                    thal]])  
    prediction = model.predict(input_cols)
    output = round(prediction[0])
    return render_template("index.html", prediction_text='heart disease prediction = {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)
   