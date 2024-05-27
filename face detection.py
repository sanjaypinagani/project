 
 #Unzipping the file
!tar -xf /content/drive/MyDrive/fer2013.tar.gz
 
from google.colab import drive drive.mount('/content/drive')
 
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.m
 
import tensorflow.compat.v1 as tf import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D from keras.layers import Dense, Activation, Dropout, Flatten
 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
 import numpy as np import matplotlib.pyplot as plt  
#-----------------------------#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) #max: 1 gpu, 56 cpu sess = tf.Session(config=config) tf.keras.backend.set_session(sess) #-----------------------------#variables num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral batch_size = 128 epochs = 30
#------------------------------
#read kaggle facial expression recognition challenge dataset (fer2013.csv)
#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognit
 
 with open("fer2013/fer2013.csv") as f:
  content = f.readlines()  
lines = np.array(content)  num_of_instances = lines.size
print("number of instances: ",num_of_instances) print("instance length: ",len(lines[1].split(",")[1].split(" ")))  
#-----------------------------#initialize trainset and test set x_train, y_train, x_test, y_test = [], [], [], []  
#-----------------------------#transfer train and test set data for i in range(1,num_of_instances):
    try:
emotion img usage = lines[i] split(" ")
        emotion, img, usage = lines[i].split( , )
                  val = img.split(" ")
                    pixels = np.array(val, 'float32')
                emotion = keras.utils.to_categorical(emotion, num_classes)
            if 'Training' in usage:             y_train.append(emotion)             x_train.append(pixels)         elif 'PublicTest' in usage:             y_test.append(emotion)             x_test.append(pixels)     except:
      print("", end="")
 
#-----------------------------#data transformation for train and test sets x_train = np.array(x_train, 'float32') y_train = np.array(y_train, 'float32') x_test = np.array(x_test, 'float32') y_test = np.array(y_test, 'float32')
 
x_train /= 255 #normalize inputs between [0, 1] x_test /= 255
 x_train = x_train.reshape(x_train.shape[0], 48, 48, 1) x_train = x_train.astype('float32') x_test = x_test.reshape(x_test.shape[0], 48, 48, 1) x_test = x_test.astype('float32')
 print(x_train.shape[0], 'train samples') print(x_test.shape[0], 'test samples')
#-----------------------------#construct CNN structure model = Sequential()  #1st convolution layer model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1))) model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 #2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu')) model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 #3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu')) model.add(Conv2D(128, (3, 3), activation='relu')) model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
model.add(Flatten())  #fully connected neural networks model add(Dense(1024 activation='relu')) model.add(Dense(1024, activation= relu )) model.add(Dropout(0.2)) model.add(Dense(1024, activation='relu')) model.add(Dropout(0.2))
 
model.add(Dense(num_classes, activation='softmax'))
#-----------------------------#batch process gen = ImageDataGenerator() train_generator = gen.flow(x_train, y_train, batch_size=batch_size)  #----------------------------- 
model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)
 #----------------------------- 
fit = True
 if fit == True:
  #model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset   model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #train f else:   model.load_weights('/data/facial_expression_model_weights.h5') #load weights
 
  #------------------------------
""" #overall evaluation
score = model.evaluate(x_test, y_test) print('Test loss:', score[0]) print('Test accuracy:', 100*score[1]) """
#------------------------------
#function for drawing bar chart for emotion preditions def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')     y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)     plt.xticks(y_pos, objects)     plt.ylabel('percentage')     plt.title('emotion')
    
    plt.show() #------------------------------
number of instances:  35888 instance length:  2304 
28709 train samples 
3589 test samples 
Epoch 1/30 
/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/training.py:
  warnings.warn('`Model.fit_generator` is deprecated and ' 
128/128 [==============================] - 3s 18ms/step - loss: 1.8354 - accuracy: Epoch 2/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.8002 - accuracy: Epoch 3/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.7212 - accuracy: Epoch 4/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.6407 - accuracy: Epoch 5/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.5693 - accuracy: Epoch 6/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.5145 - accuracy: Epoch 7/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.4663 - accuracy: Epoch 8/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.4364 - accuracy: Epoch 9/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.3935 - accuracy: Epoch 10/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.3662 - accuracy: Epoch 11/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.3187 - accuracy: Epoch 12/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.3135 - accuracy: Epoch 13/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.2912 - accuracy: Epoch 14/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.2637 - accuracy: Epoch 15/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.2334 - accuracy: Epoch 16/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.2108 - accuracy: Epoch 17/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.1820 - accuracy: Epoch 18/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.1789 - accuracy: Epoch 19/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.1414 - accuracy: Epoch 20/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.1419 - accuracy: Epoch 21/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.1205 - accuracy: Epoch 22/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.1134 - accuracy: Epoch 23/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.0699 - accuracy: Epoch 24/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.0831 - accuracy: Epoch 25/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.0679 - accuracy: Epoch 26/30 
128/128 [==============================] - 2s 18ms/step - loss: 1.0334 - accuracy: 
model.save('model25.h5')
#Evaluation
train_score = model.evaluate(x_train, y_train, verbose=0) print('Train loss:', train_score[0]) print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(x_test, y_test, verbose=0) print('Test loss:', test_score[0]) print('Test accuracy:', 100*test_score[1])
Train loss: 0.9762634038925171 
Train accuracy: 63.185763359069824 
Test loss: 1.1882857084274292 Test accuracy: 55.19643425941467 
#Confusion Matrix.  from sklearn.metrics import classification_report, confusion_matrix
 pred_list = []; actual_list = [] predictions = model.predict(x_test) #predictions=classifier.predict(x_test) for i in predictions:
   pred_list.append(np.argmax(i))  for i in y_test:
   actual_list.append(np.argmax(i))  confusion_matrix(actual_list, pred_list)
array([[237,   8,  21,  46,  88,  12,  55],        [ 21,  17,   3,   1,  10,   0,   4], 
       [ 96,   6,  99,  46, 122,  41,  86], 
       [ 39,   3,  22, 731,  40,   5,  55], 
       [ 97,   6,  46,  69, 293,   9, 133], 
       [ 28,   0,  39,  36,  18, 279,  15], 
       [ 64,   4,  24,  81, 100,   9, 325]])
monitor_testset_results = True
 if monitor_testset_results == True:   #make predictions for test set   predictions = model.predict(x_test)    index = 0   for i in predictions:     if index < 30 and index >= 20:
      #print(i) #predicted scores
      #print(y_test[index]) #actual scores
            testing_img = np.array(x_test[index], 'float32')       testing_img = testing_img.reshape([48, 48]);
            plt.gray()       plt.imshow(testing_img)       plt.show()             print(i)       emotion_analysis(i)       print("----------------------------------------------")     index = index + 1 
 
[0.01701456 0.00081983 0.05002777 0.8089382  0.02674721 0.0203561  0.07609632] 
 
---------------------------------------------- 
 
[4.0989071e-02 7.4134913e-04 1.2261624e-02 1.4552251e-02 7.5934818e-03  8.9145392e-01 3.2408342e-02] 
 
 
---------------------------------------------- 
 
[0.06298877 0.0038647  0.2408447  0.02466396 0.5560561  0.00702478  0.10455696] 
 
---------------------------------------------- 
 
[0.07373388 0.00168616 0.15044174 0.03999103 0.5522884  0.00532372  0.17653504] 
 
 
---------------------------------------------- 
 
[0.24244437 0.00632507 0.08277731 0.3393256  0.2600014  0.01133873  0.05778749] 
 
---------------------------------------------- 
 
[0.12625332 0.00646819 0.09299825 0.05951758 0.26059163 0.01503523  0.43913573] 
 
 
---------------------------------------------- 
 
[0.05959696 0.03893325 0.27407566 0.14596546 0.3627533  0.00710405  0.11157135] 
 
---------------------------------------------- 
 
 
[3.4881439e-02 6.7239511e-05 3.8950968e-01 4.5537315e-02 1.8629777e-01  8.4756248e-02 2.5895035e-01] 
 
---------------------------------------------- 
 
[8.7088126e-01 3.5084732e-04 4.5998320e-02 2.4184503e-03 3.8617119e-02 
def emotion_analysis 7.8436406e-03 3.3890415e-02](emotions):	 
    
    
 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator ---------------------------------------------- 
 import numpy as np
import matplotlib.pyplot as plt
  file = 'capture.jpg' true_image = image.load_img(file) img = image.load img(file, grayscale=True, target size=(48, 48)) img  image.load_img(file, grayscale True, target_size (48, 48))
 
 x /= 255
 
[4.7101988e-04 2.1764838e-06 4.0034596e-02 2.2320568e-03 7.8559440e-04 
custom = model.predict(x)
 9.5568734e-01 7.8714977e-04] emotion_analysis(custom[0])
 
 plt.gray() plt.show()
import cv2             def facecrop(image):  
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +facedata)
 
    img = cv2.imread(image)
 
    try:             minisize = (img.shape[1],img.shape[0])         miniframe = cv2.resize(img, minisize)          faces = cascade.detectMultiScale(miniframe)
         for f in faces:
            x, y, w, h = [ v for v in f ]             cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
             sub_face = img[y:y+h, x:x+w]                          cv2.imwrite('capture.jpg', sub_face)
            #print ("Writing: " + image)
     except Exception as e:
        print (e)  
    #cv2.imshow(image, img)
  if __name__ == '__main__':     facecrop('photo.jpg')
from IPython.display import display, Javascript from google.colab.output import eval_js from base64 import b64decode
 def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''     async function takePhoto(quality) {       const div = document.createElement('div');       const capture = document.createElement('button');       capture.textContent = 'Capture';       div.appendChild(capture);        const video = document.createElement('video');       video.style.display = 'block';       const stream = await navigator.mediaDevices.getUserMedia({video: true});
       document.body.appendChild(div);       div.appendChild(video);       video.srcObject = stream;       await video.play();
 
      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);  
      // Wait for Capture to be clicked.       await new Promise((resolve) => capture.onclick = resolve);
       const canvas = document.createElement('canvas'); canvas width = video videoWidth;
 
