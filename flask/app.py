
from flask import Flask, current_app, send_file, request
app = Flask(__name__, static_url_path='')

import io
import numpy as np

from test import evaluate, to_image

@app.route('/generate', methods = ['POST'])
def generate():
    # labelmap = np.load('/app/labelmap.npy')
    labelmap = np.asarray(request.json)

    image = evaluate(labelmap)
    image = to_image(image)

    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    image.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
