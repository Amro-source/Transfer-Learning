# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:51:06 2020

@author: Zikantika
"""

# example of using the vgg16 model as a feature extraction model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump

#import tensorflow as tf 
import tensorflow as tf

#
#tf.lite.TFLiteConverter(
#    funcs, trackable_obj=None
#)


# load an image from file
image = load_img('dog.1048.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# load model
model = VGG16()
# remove the output layer
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)


## Converting a tf.Keras model to a TensorFlow Lite model.
#converter = lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#tflite_model = converter.convert()
#open("converted_model.tflite", "wb").write(tflite_model)



converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


#if you want to save the TF Lite model use below steps or else skip
#tflite_model_files = pathlib.Path('K:\dogs-vs-cats')
#tflite_model_file.write_bytes(tflite_model)



# summarize the model
model.summary()
# get extracted features
features = model.predict(image)
print(features.shape)
# save to file
dump(features, open('dog.pkl', 'wb'))



#def export_tflite(classifier):
#        with tf.Session() as sess:
#            # First let's load meta graph and restore weights
#            latest_checkpoint_path = classifier.latest_checkpoint()
#            saver = tf.train.import_meta_graph(latest_checkpoint_path + '.meta')
#            saver.restore(sess, latest_checkpoint_path)
#
#            # Get the input and output tensors
#            input_tensor = sess.graph.get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
#            out_tensor = sess.graph.get_tensor_by_name("dnn/logits/BiasAdd:0")
#
#            # here the code differs from the toco example above
#            sess.run(tf.global_variables_initializer())
#            converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor], [out_tensor])
#            tflite_model = converter.convert()
#            open("converted_model.tflite", "wb").write(tflite_model)