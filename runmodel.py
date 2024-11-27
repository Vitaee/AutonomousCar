from utils import preProcessing
import socketio
from flask import Flask
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys
#socket connect
socketIO = socketio.Server(cors_allowed_origins="*")

app = Flask(__name__)

maxSpeed = 10



@socketIO.on('connect')
def connect(sid, environ):
    print("Bağlantı sağlandı.")
    print("Connected:", sid)
    return True

    
# telemetry / watch simulator
@socketIO.on('telemetry')
def telemetry(sid, data):

    speed = float(data['speed'].replace(',','.'))  
    img = data['image']
    img = Image.open( BytesIO( base64.b64decode( img) ))
    img = np.asarray(img)
    img = preProcessing(img)
    img = np.array( [img] )
    
    steering = float( model.predict(img) )
    throttle = 1.0 - speed / maxSpeed
    print('{} , {}, {}'.format(steering, throttle, speed))
    sendCommand(steering, throttle)

import eventlet
# send command
def sendCommand(steering, throttle):
    print('Gönderilen: ', steering, throttle)
    socketIO.emit('steer',data={'steering_angle':steering.__str__(),
                               'throttle':throttle.__str__()})
    
    
if __name__ == '__main__':
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError

    mse_loss = MeanSquaredError()  # Define the MSE function explicitly
    model = load_model('bestmodel.h5', custom_objects={'mse': mse_loss})
    
    app = socketio.Middleware(socketIO, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
