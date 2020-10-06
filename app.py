from flask import Flask
from firebase import firebase
import cv2
import numpy as np
import os
import joblib
import urllib.request

#the brain of the mighty Project
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
    [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#load fireabse realtime databse link
firebase = firebase.FirebaseApplication("https://_firebase app name_.firebaseio.com/", None)

app = Flask(__name__)
@app.route('/home/<pos>')
def classify(pos):
    #get the link of the image from firebase realtime Database
    string  = '%s' % pos
    data = firebase.get('/'+string, '')
    link = data["url"]
    
    #open the image using urllib.requests and convert the same into
    #an array as opencv reads the image as an array 
    req = urllib.request.urlopen(link)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    #reshape the array
    img = cv2.imdecode(arr, -1)
    
    #load the model using joblib
    model = joblib.load(os.path.abspath(os.path.dirname(__file__).replace("",""))+"/assets/model.sav")
    #get the histogram 
    histt = extract_color_histogram(img)
    histt2 = histt.reshape(1, -1)
    #predict
    prediction = model.predict(histt2)
    acc = mse( , prediction)
    #comapare the output to return the corresponding class
    if prediction == [1]:
        return "Pneumonia"
    else:
        return "Normal"
    
if __name__ == '__main__':
    app.run(debug=True) 
