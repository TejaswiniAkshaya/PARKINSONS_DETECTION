from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
app=Flask(__name__)
model1=pickle.load(open('main2.pkl','rb'))      

@app.route("/")
def map():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def home():
    name=request.form['name']
    phone=request.form['phone']
   
    d1=request.form['b']   
    d2=request.form['c']
    d3=request.form['d']
    d4=request.form['e']
    d5=request.form['f']
    d6=request.form['g']
    d7=request.form['h']
    d8=request.form['i']
    d9=request.form['j']
    d10=request.form['k']
    d11=request.form['l']
    d12=request.form['m']
    d13=request.form['n']
    d14=request.form['o']
    d15=request.form['p']
    d16=request.form['q']
    d17=request.form['r']
    d18=request.form['s']
    d19=request.form['t'] 
    d20=request.form['u']
    d21=request.form['v']
    d22=request.form['w']
    fullname=name
    words=fullname.split()

    inp=[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10
         ,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
         d21,d22]
    try:
      
      arr=np.asarray(inp)
      arr1=arr.reshape(1,-1)
      scaler = joblib.load('scaler.pkl')
      std=scaler.transform(arr1)
      prediction=model1.predict(std)
      print(prediction)
      d=prediction[0]
      return render_template('prediction.html',data=d,name=words[0])
    except ValueError as e:
      return render_template('prediction.html',data=2)
    

   
if __name__=="__main__":
    app.run(debug=True)



