import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize


modelF = load_model('scmbK.h5')

def processimg(img):
   img = resize(img,(224,224,3), mode = "constant",anti_aliasing=True)
   img.astype(np.float32)
   img = cv2.resize(img,(224,224))     # resize image to match model's expected sizing
   img = img.reshape(1,224,224,3)
   img=img.astype('float32')
   img = img/255
   return img

def predict(img_path):
  img = cv2.imread(img_path)
  img = processimg(img)
  w = modelF.predict_classes(img)
  wp = modelF.predict_proba(img)
  result =[]
  if w[0] == 0:
    result.append('Benign cancer') 
  elif w[0] == 1:
    result.append('Malignant cancer')
  Cscores = []
  if wp[0][0]>wp[0][1]:
     Cscores.append(round(wp[0][0]*100))
  else:
     Cscores.append(round(wp[0][1]*100))
  Output = []
  Output.append('{} confidence score is {}%'.format(result[0], Cscores[0]))
  return Output
