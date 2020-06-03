style_weight=1
total_variation_weight=1.5
norm_term = 6
norm_weight = 0.1
epochs = 30
steps_per_epoch = 100
history_loss = []
from utils import *
from custom_vgg import *
import argparse
import os
outPath = None
style_Path = None

def style_loss(outputs):
    style_outputs = outputs['style']###输入的图像结果
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    loss = style_loss
    return loss #纹理风格损失
def noise_loss(X):
    return (norm_loss(X)**norm_term)*norm_weight
def vgg_layers(layer_names,vgg):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model
def parse_args():
    global outPath,style_Path

    parser = argparse.ArgumentParser()
    parser.add_argument("texture",
        help="path to the image you'd like to resample")
    parser.add_argument("--output",
        default=outPath,
        help="path to where the generated image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    style_path = os.path.realpath(args.texture)
    outPath = os.path.realpath(args.output)


class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers, model):
        super(StyleModel, self).__init__()
        self.vgg = vgg_layers(style_layers, model)
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs = (outputs[:self.num_style_layers])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'style': style_dict}
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        #loss = style_loss(outputs) + noise_loss(image)
        #loss += total_variation_weight * total_variation_loss(image)
        loss = style_loss(outputs)
        history_loss.append(loss)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
if __name__ == '__main__':
    parse_args()
    custom_vgg = creatModel()
    style_layers = [layer.name for layer in custom_vgg.layers][1:]
    num_style_layers = len(style_layers)
    style_image = load_img(style_Path)
    image = tf.Variable(tf.random.normal(shape=style_image.shape, mean=0, stddev=1))  # 用纹理图像生成初始噪声图像
    style_extractor = vgg_layers(style_layers, custom_vgg)
    style_outputs = style_extractor(style_image * 255)
    extractor = StyleModel(style_layers, custom_vgg)
    style_targets = extractor(style_image)['style']
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    for n in range(epochs):
      for m in range(steps_per_epoch):
        train_step(image)
    outfile = tensor_to_image(image)
    outfile.save(outPath)





