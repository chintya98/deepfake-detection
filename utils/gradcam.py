import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
  grad_model = tf.keras.models.Model(
    [model.inputs], 
    [model.get_layer(last_conv_layer_name).output, model.output]
  )
  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]
  grads = tape.gradient(loss, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
  return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
  heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
  superimposed_img = heatmap_colored * alpha + img
  superimposed_img = np.uint8(superimposed_img)
  return superimposed_img
