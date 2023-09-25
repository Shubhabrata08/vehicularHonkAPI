from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import tensorflow as tf
import os
from vehicularHonk.settings import BASE_DIR
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model
from rest_framework import status
from .models import ImageUploadSerializer  # Import your serializer
from PIL import Image
import numpy as np
# Create your views here.

def createModel():
    image_size = 224  # We'll resize input images to this size
    num_patches = 8
    patch_size =224//num_patches   # Size of the patches to be extract from the input images
    input_shape=(224,224,3)
    # Input dimensions
    input_shape = (224, 224, 3)
    # Patch size
    # Number of classes
    num_classes = 4

    # Input layer
    inputs = Input(shape=input_shape)

    # Divide the image into patches
    patches = tf.image.extract_patches(images=inputs,
                                sizes=[1, image_size//patch_size, image_size, 1],
                                strides=[1,image_size//patch_size, image_size, 1],
                                rates=[1, 1, 1, 1],
                                padding='VALID')

    # Reshape the patches to have 3 channels
    patches = tf.reshape(patches, [-1, patch_size, image_size, 3*(image_size//patch_size)])
    print(patches.shape)

    # CNN for patch processing
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(patches)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    flat = Flatten()(pool2)
    fc1 = Dense(128, activation='relu')(flat)
    drop=Dropout(0.5)(fc1)
    outputs = Dense(num_classes,activation="softmax")(drop)


    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    # Compile the model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics=["accuracy"]
    )
    model.load_weights(os.path.join(BASE_DIR,f"vehicularHonkAPI\patchedModels\patched{num_patches}weights.h5"))
    return model

model=createModel()


@api_view(['POST'])
def predictHonk(request):
    serializer = ImageUploadSerializer(data=request.data)
    if serializer.is_valid():
        print(request.data['image'])
        receivedImage=request.FILES.get('image')
        renderedImg=Image.open(receivedImage)
        renderedImg.save('image.jpg')
        modelInput=np.array(renderedImg)
        print(modelInput.shape)
        prediction=model.predict(np.array([modelInput]))
        audioClass=-1
        for i in range(len(prediction[0])):
            if prediction[0][i]==1:
                audioClass=i
        return JsonResponse({"state":str(audioClass)})
    return JsonResponse({"state":"Failure"})
    

