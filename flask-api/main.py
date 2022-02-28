from platform import python_version
from flask import Flask, render_template
from flask import request
import json
import sys
import os
import datetime

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import librosa
import io
import numpy as np
import pandas as pd
from google.cloud import datastore

from AudioProcessing import AudioProcessing

# keras model path to load
model_path = '/workspace/assets/cnn-model.h5'
# size to the length that the model was trained at
resize_to = 9720
# n_mfcc that the model was trained at
n_mfcc = 128
# load model
model = models.load_model(filepath=model_path, compile=True)

# implement datastore
datastore_client = datastore.Client()


def store_time(dt):
    entity = datastore.Entity(key=datastore_client.key('visit'))
    entity.update({'timestamp': dt})
    datastore_client.put(entity)


def fetch_times(limit):
    query = datastore_client.query(kind='visit')
    query.order = ['-timestamp']
    times = query.fetch(limit=limit)
    return times


app = Flask(__name__)


@app.route('/')
def root():
    # Store the current access time in Datastore.
    store_time(datetime.datetime.now(tz=datetime.timezone.utc))

    # Fetch the most recent 10 access times from Datastore.
    times = fetch_times(10)

    return render_template('index.html', times=times)


@app.route('/about/', methods=['GET', 'POST'])
def about():
    # python version
    python_major_ver, python_minor_ver, python_micro_ver, python_releaselevel, python_serial = sys.version_info
    python_version = str(python_major_ver) + '.' + \
        str(python_minor_ver) + '.' + str(python_micro_ver)
    # dependencies
    tf_version = str(tf.__version__)
    keras_version = str(keras.__version__)
    librosa_version = str(librosa.__version__)
    pandas_version = str(pd.__version__)
    numpy_version = str(np.__version__)

    # model
    model_loaded = False
    try:
        n_layers = len(model.layers)
        model_loaded = True
    except Exception:
        model_loaded = False

    return json.dumps({'python_version': python_version,
                       'Working Directory': os.getcwd(),
                       'File_And_Objects': str(os.listdir()),
                       'modle_downloaded': str(os.path.isfile(model_path)),
                       'tf_version': tf_version,
                       'keras_version': keras_version,
                       'librosa_version': librosa_version,
                       'pandas_version': pandas_version,
                       'numpy_version': numpy_version,
                       'model': model_loaded}), 200


@app.route('/evaluatecough/', methods=['GET', 'POST'])
def evaluate_cough():
    data = request.get_data()
    data_length = request.content_length

    if (data_length > 1024 * 1024 * 10):
        return json.dumps({'error': 'File is to large, size={}'.format(data_length)})

    tmp = io.BytesIO(data)
    f = AudioProcessing(file_path=tmp, resize_to=resize_to)
    spectrogram = f.spectrogram()
    mfccs_gen = f.get_mfccs_generator(spectrogram=spectrogram, n_mfcc=n_mfcc)

    # predict cough type
    prediction = model.predict(mfccs_gen)
    # print(prediction)
    percent_dry = prediction[0, 0]
    percent_dry = round(percent_dry * 100, 2)
    percent_wet = prediction[0, 1]
    percent_wet = round(percent_wet * 100, 2)
    prediction_class = prediction[0].argmax(axis=-1)
    if prediction_class == 0:
        prediction_class = 'dry'
    else:
        prediction_class = 'wet'

    return json.dumps({'percent_dry': str(percent_dry),
                       'percent_wet': str(percent_wet),
                       'class': prediction_class}), 200


tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
