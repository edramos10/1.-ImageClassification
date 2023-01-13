import cv2
import numpy as np
from PIL import Image
from keras import models
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
#Load the saved model

#'7b806db3818e4b25b74780c23c93082f'
import mlflow
#from mlflow_extend import mlflow
mlflow.set_tracking_uri("http://localhost:1234")
EXPERIMENT_NAME='KEYPADSv1'#'FKL-Faceplates-kfolded-092822'
logged_model ='C:/Users/anavarro4/Anaconda3/envs/mlflow/Scripts/artifacts/9/c0575ba2468f44f79c249074f9a76a25/artifacts/model' #'./artifacts/10/5ee87181d37f4d0db2b16396851c6d10/artifacts/model'#'./artifacts/10/642f76d8d61c463da8c35ccbad1207c6/artifacts/model'#'./artifacts/2/610c4e76b446444d82ae4f2a0b25da43/artifacts/model' #'./artifacts/10/02a53a696ed94df899bced0675a28d6a/artifacts/model'  #'./artifacts/5/7b806db3818e4b25b74780c23c93082f/artifacts/model'#'runs:/d39ccd3aac214bd4bcc535c6ce83cf74/model' #'runs:/a4229a4d1422454cbd17f97cfdc5b944/MODEL' #'runs:/ec1c138c5def45d2a01cc9e4ea53fd44/model'
#logged_model=r'\\allegion\americas\ens/groups/Public/DeepLearning/Models/bdcb7b1d86144f5baf321266a7970f45/artifacts/model'
#'./artifacts/3/d39ccd3aac214bd4bcc535c6ce83cf74/artifacts/model'


# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
MODEL=loaded_model

#MODEL = tf.compat.v1.keras.experimental.load_from_saved_model(artifact_dir)#tf.keras.models.load_model(artifact_dir)
#MODEL = tf.keras.models.load_model('model/model-best-fiery-night-11.h5') #('model/model-best-efficient-snowflake.h5', custom_objects={'HammingLoss':hamming_loss,'cohen_kappa':cohen_kappa})#'model/model-best-InceptionV3-09072022.h5')

# Get the input shape for the model layer
input_shape = (1,299,299,3)#MODEL.layers[0].input_shape

model = MODEL#models.load_model('model.h5')
video = cv2.VideoCapture(1)
# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,450)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2



while True:
        _, frame = video.read()
      
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame, 'RGB')

        im = im.resize((299,299))
        img_array = np.array(im)
 
        img_array = np.expand_dims(img_array, axis=0)
        class_names=["PCB1", "PCB2","Unknown"]
     
        img_array=img_array/255
        #img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        prediction = predictions[0]
        likely_class = np.argmax(prediction)
        
        class_name_predicted=class_names[likely_class]
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2BGR)
        cv2.putText(frame,str(round((prediction[likely_class]*100),2))+"%" +"/"+ class_name_predicted, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()