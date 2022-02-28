#####
# !/Users/christopherfesta/opt/anaconda3/envs/tf/bin python

from crypt import methods
from flask import Flask
from flask import request
from flask import Response
#from flask_cors import CORS
from pprint import pprint
import json
import uuid

import logging
import tensorflow as tf
from tensorflow.keras import models
import librosa
import pandas as pd
import numpy as np

from AudioProcessing import AudioProcessing
from Dataset import SingleInputGenerator

# suppress warrnings
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
print('Here')

# CORS(app)
app = Flask(__name__)

if __name__ == 'main':
    model_path = './assets/cnn-model.h5'
    save_dir = './audio_files/'
    resize_to = 9720  # size to the length that the model was trained at
    n_mfcc = 128
    model = models.load_model(filepath=model_path, compile=True)
    print('Loaded keras model')

    app.run(host='0.0.0.0', port=105)


@app.route('/audio', methods=['POST'])
def process_audio():
    data = request.get_data()
    data_size = request.content_length

    if data_size > 1024 * 1024 * 10:
        return 'File too large!', 400

    # process audio file
    print('Processing data', data)
    # Save audio file
    f = request.files['file']
    file_path = save_dir + str(uuid.uuid4)
    f.save(file_path)

    f = AudioProcessing(file_path=file_path, resize_to=resize_to)
    spectrogram = f.spectrogram()
    mfccs = librosa.feature.mfcc(S=spectrogram, n_mfcc=n_mfcc)
    row = [[0, mfccs]]
    df = pd.DataFrame(row, columns=['file_num', 'mfccs'])

    predict_mfccs = np.array([df['mfccs'].iloc[i]
                              for i in range(len(df))], dtype='object')
    predict_mfccs = predict_mfccs.reshape(
        predict_mfccs.shape[0], predict_mfccs.shape[1], predict_mfccs.shape[2], 1)

    # generate predictor data, using dummy class
    predictor_gen = SingleInputGenerator(
        X1=predict_mfccs, Y=[1], batch_size=1, n_classes=2, shuffle=False)

    # predict cough type
    prediction = model.predict(predictor_gen)
    # print(prediction)
    prediction_percent_dry = prediction[0, 0]
    prediction_percent_wet = prediction[0, 1]
    prediction_class = prediction[0].argmax(axis=-1)
    if prediction_class == 0:
        prediction_class = 'dry'
    else:
        prediction_class = 'wet'

    print('Completed processing:', prediction_percent_dry,
          prediction_percent_wet, prediction_class)

    return json.dumps({'prediction_percent_dry': str(prediction_percent_dry),
                       'prediction_percent_wet': str(prediction_percent_wet),
                       'prediction_class': str(prediction_class)}), 200
