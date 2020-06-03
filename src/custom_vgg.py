import tensorflow as tf
import  numpy as np
def setAvePolConfig(configDict):
  configDict['class_name']= 'AveragePooling2D'
  configDict['padding'] = 'same'
def creatModel():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    config = vgg.get_config()
    layersConfig = config['layers']
    pool_index=  [3,6,11,16,21]
    for i in pool_index:
      setAvePolConfig(layersConfig[i])
    model = tf.keras.Model.from_config(config)
    model.set_weights(vgg.get_weights())

    return model