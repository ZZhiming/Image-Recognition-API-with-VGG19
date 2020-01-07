import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pylab as plt
import numpy as np
import tensorflow_hub
import PIL.Image
import cv2

vgg_model = tf.keras.models.load_model("vgg.h5")

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def crop_center(image):
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
    return image

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

# Use pretrained VGG model to classify images
def predict(img):
    x = tf.keras.applications.vgg19.preprocess_input(img)
    x = tf.image.resize(x, (224, 224))
    prediction_probabilities = vgg_model(x.numpy().reshape(1,224,224,3))

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    return [{class_name: str(prob)} for (number, class_name, prob) in predicted_top_5]
