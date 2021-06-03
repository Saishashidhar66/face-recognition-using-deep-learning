from keras.layers import Input  # is used to instantiate a Keras tensor
from keras.layers import Lambda # Lambda is used to transform the input data using an expression or function
from keras.layers import Dense # dense is the final layer
from keras.layers import Flatten # to flatten the inpunt and convert single linear vector
from keras.models import Model # Model groups layers into an object with training and inference features.
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input #Preprocesses a tensor or Numpy array encoding a batch of images.
from keras.preprocessing import image # we work on image so image
from keras.preprocessing.image import ImageDataGenerator #to rescale images  image data agumentation
from keras.models import Sequential # basic NN 
import numpy as np 
from glob import glob #
import matplotlib.pyplot as plt #pllot is predectioons

image_size =[224,224]

train_path = r'E:\facerec-using-TLM\images\training/'
valid_path = r'E:\facerec-using-TLM\images\validation/'
# add preprocessing layer  to the vgg
IMAGE_SIZE = [224, 224]


# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('images/training/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('images/training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('images/validation',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

model.save(r'E:\facerec-using-TLM/facefeatures_new_modelnew.h5')
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')



