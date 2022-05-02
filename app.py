import joblib
from flask import Flask, render_template, request
import cv2
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/')
def show_predict_stock_form():
	return render_template('predictorform.html')


@app.route('/result', methods=['POST'])	
def results():
	form = request.form
	if request.method == 'POST':
		model = joblib.load("jaimin_nmuna.pkl")
		image = form['xray-image']
		img = cv2.imread(image, 0)
		cv2.imwrite('./static/temp/temp_saved.jpg', img)
		normalized_image = cv2.resize(img, (64, 64)).flatten()
		value = model.predict(normalized_image.reshape(1, -1))
		probability = model.predict_proba(normalized_image.reshape(1, -1))[0][1]
		response = ""
		if value[0] == 1.0:
			response = "Covid Detected"
		else:
			response = "Covid Not Detected"
		return render_template('resultsform.html', image=form['xray-image'], response=response, probability=(probability*100))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
# app.run("localhost", "9999", debug=True)
