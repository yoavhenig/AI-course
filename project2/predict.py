import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model('./model1.h5', compile=False)

def process_image(image):
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image = image/255.0
    return image


def predict_class(image):
    result_prob = model.predict(image)
    result_class = model.predict_classes(image)

    prediction = result_class[0]
    percentage = (result_prob[0][prediction])

    return prediction, percentage

#image = load_img('sample_image.png',grayscale=True, target_size=(28,28))
#image = process_image(image)
#pre_class, pre_prob = predict_class(image)
#print('Classified as class: {pre_class} with propability of: {a:.6f}% ')