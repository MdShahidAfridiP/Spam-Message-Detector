from flask import Flask,render_template,url_for,request
from tensorflow.keras.models import load_model
import pickle
app=Flask(__name__)

model=load_model("spam_model.h5")
tfidf=pickle.load(open("transform.pkl","rb"))

@app.route("/")
def home():
    return render_template("spam.html")

@app.route("/predict",methods=['POST'])
def predict():
    prediction=0
    message=request.form['message']
    message=tfidf.transform([message]).toarray()
    result=model.predict(message)
    if (result[0][0])>=0.5:
        prediction=1
    else:
        prediction=0
    return render_template("spam.html",prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)