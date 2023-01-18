#Allegion 2022
#Author: Edgar Ramos - MLOPS
#DevOps Engineer: F. Moreno
#Date: 09-07-22
#This service is utilized to perform inference over ADCO baseplates

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.keras.backend.clear_session()  # Para restablecer fácilmente el estado del portátil.

from fastapi import FastAPI,File, UploadFile

from pydantic import BaseModel
import cv2 as cv
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
# Imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from tensorflow.keras.models import load_model
from typing import List
import io
import sys
from tensorflow.keras.models import load_model

#import tensorflow_addons as tfa


from mlflow_extend import mlflow
mlflow.set_tracking_uri("http://localhost:1234")
EXPERIMENT_NAME='ADCO-baseplates-kfolded-09242022'


#wandb artifacts
import wandb
run = wandb.init()
artifact = run.use_artifact('aimfg-california/Nigel-Baseplates-2022/model-rosy-meadow-247:v0', type='model')
artifact_dir = artifact.download()

logged_model_wandb =artifact_dir#'./artifacts/rosy-meadow-247/model-best.h5'




# Load model as a PyFuncModel if MLFLOW is utilized

#loaded_model = mlflow.pyfunc.load_model(logged_model)
#MODEL=loaded_model
MODEL=load_model(logged_model_wandb)


# Get the input shape for the model layer
input_shape = (1,299,299,3)#MODEL.layers[0].input_shape

app = FastAPI()

class UserInput(BaseModel):
    user_input: float
# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: int
  class_name_predicted: str

@app.get('/')
async def index():
    return {"Message": "This is Index"}



# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)

async def prediction_route(file: UploadFile = File(...)):

  # Ensure that this is an image
  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    image_in_Bytes = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
 
    img = image_in_Bytes.resize((299,299))
    # Convert from RGBA to RGB *to avoid alpha channels*
    if img.mode == 'RGBA':
      img = img.convert('RGB')

    # Convert image into grayscale *if expected*
    #if input_shape[3] and input_shape[3] == 1:
    #  pil_image = pil_image.convert('L')
    img = img.convert('RGB')

    # Convert image into numpy format
    numpy_image = np.array(img)

    # Scale data (depending on your model)
    numpy_image = np.expand_dims(numpy_image, axis=0)
    #numpy_image = preprocess_input(numpy_image)
    numpy_image = numpy_image/255
    class_names=["AD_CYL", "AD_MS","CO_CYL", "CO_MS", "EXIT", "TRAY_ONLY"]
    
    predictions = MODEL.predict(numpy_image)
    
    prediction = predictions[0]
    likely_class = np.argmax(prediction)
    class_name_predicted=class_names[likely_class]


    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction.tolist(),
      'likely_class': likely_class,
      'class_name_predicted': class_name_predicted
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))

async def predict(UserInput: UserInput):

    prediction = MODEL.predict([UserInput.user_input])

    return {"prediction": float(prediction)}