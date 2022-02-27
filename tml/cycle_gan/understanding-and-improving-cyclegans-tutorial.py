# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] papermill={"duration": 0.029652, "end_time": "2020-09-12T13:34:27.252152", "exception": false, "start_time": "2020-09-12T13:34:27.222500", "status": "completed"} tags=[]
# # How to Paint with MORE Generative Adversarial Networks
#
# *This is part 2, see part 1 [here](https://www.kaggle.com/jesperdramsch/how-to-paint-with-gans-least-squares-gan-starter)*
#
# You can create Monets with Generative Adversarial Networks (GAN) in a few different ways. We can generate them from scratch using one GAN, where the GAN basically imagines a Monet from scratch. GANs are technically two networks that work against each other, illustrated below. The artist (generator) draws its inspiration from a noise sample and creates a rendering of the data you are trying to generate with said GAN. The private investigator (discriminator) randomly gets assigned real and fake data to investigate. 
#
# ![](https://media1.tenor.com/images/4c6c1a33f7e10573c7c9a5d574de417a/tenor.gif?itemid=8224930)
#
# Because us machine learning scientists had a bit too much GPUs running idle, people came up with the idea of cycle-consistent GANs. CycleGANs were introduced for [unpaired Image-to-Image Translation](https://junyanz.github.io/CycleGAN/), for when you don't have Monet available to paint your favorite subject. They're pretty useful generally and have been applied in many domains and style transfer applications. The problem?! This is now two GANs to train that perforn the forward and backward transformation to create the new style from nothing. Finally we can compare apples to oranges.
#
# ![](https://junyanz.github.io/CycleGAN/images/objects.jpg)
#
# In this tutorial we'll look in-depth into:
#
# - Data Augmentation
# - Underlying Neural Network Architectures
# - CycleGAN architectures
# - Better Loss functions
#
# Bonus? Their failure modes are hilarious:
#
# ![](https://camo.githubusercontent.com/757b691307b52fe8a0806dde3a560dc068dbf5b3/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f6661696c7572655f707574696e2e6a7067)
#
# The main idea behind a CycleGAN is that two Generative Adversarial Networks are trained. Network one learning the forward transformation to the target domain and the second network learning the inverse transformation back to the original image domain. Pairing this with the GAN loss of creating "believable" images, at least according to the discriminator of the GAN, yields some surprisingly good transformations.
#
# *This copies in part from my [Intro to Deepfakes](https://www.kaggle.com/jesperdramsch/intro-to-deep-fakes-videos-and-metadata-eda) if you're interested to learn how GANs are used to alter images, videos, and sounds. (Did I mention they're quite versatile?!) This also builds on the Baseline Tutorial. [Please head over and upvote!](https://www.kaggle.com/amyjang/monet-cyclegan-tutorial)*

# + [markdown] papermill={"duration": 0.02777, "end_time": "2020-09-12T13:34:27.308798", "exception": false, "start_time": "2020-09-12T13:34:27.281028", "status": "completed"} tags=[]
# ## Introduction and Setup
#
# For this tutorial, we will be using the TFRecord dataset. Import the following packages and change the accelerator to TPU. Because TPUs are pretty awesome.
#
# ![](https://i.imgur.com/hRLjugH.png)

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" papermill={"duration": 12.372205, "end_time": "2020-09-12T13:34:39.709071", "exception": false, "start_time": "2020-09-12T13:34:27.336866", "status": "completed"} tags=[]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__)

# + [markdown] papermill={"duration": 0.02882, "end_time": "2020-09-12T13:34:39.767303", "exception": false, "start_time": "2020-09-12T13:34:39.738483", "status": "completed"} tags=[]
# # Load in the data
#
# We want to keep our photo dataset and our Monet dataset separate. First, load in the filenames of the TFRecords. We'll load both for the CycleGAN. For the first GAN we only need the Monets as training data.
#
# All the images for the competition are already sized to `256 x 256`. As these images are RGB images, set the channel to 3. Additionally, we need to scale the images to a `[-1, 1]` scale. Because we are building a generative model, we don't need the labels or the image id so we'll only return the image from the TFRecord.

# + papermill={"duration": 0.582324, "end_time": "2020-09-12T13:34:40.378518", "exception": false, "start_time": "2020-09-12T13:34:39.796194", "status": "completed"} tags=[]
GCS_PATH = KaggleDatasets().get_gcs_path()

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
print('Monet TFRecord Files:', len(MONET_FILENAMES))

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
print('Photo TFRecord Files:', len(PHOTO_FILENAMES))

# + [markdown] papermill={"duration": 0.029422, "end_time": "2020-09-12T13:34:40.437696", "exception": false, "start_time": "2020-09-12T13:34:40.408274", "status": "completed"} tags=[]
# You can see I put down a bit of augmentation using `random_jitter` and `flip` to increase our data set, because we simply don't have enough data for

# + papermill={"duration": 0.062377, "end_time": "2020-09-12T13:34:40.529500", "exception": false, "start_time": "2020-09-12T13:34:40.467123", "status": "completed"} tags=[]
IMAGE_SIZE = [256, 256]

def normalize(image):
    return (tf.cast(image, tf.float32) / 127.5) - 1

def decode_image(image):
    #image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.reshape(image, [256, 256, 3])
    image = tf.image.decode_jpeg(image, channels=3)
    #image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[256, 256, 3])
    return cropped_image

def random_jitter(image):
    # resizing to 286 x 286 x 3 
    image = tf.image.resize(image, [int(256*1.3), int(256*1.3)],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    return image

def flip(image):
    return tf.image.flip_left_right(image)

def preprocess_image_train(image, label=None):
    image = random_jitter(image)
    return image

def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image

def load_dataset(filenames, labeled=False, ordered=False, repeats=200):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.concatenate(dataset.map(flip, num_parallel_calls=AUTOTUNE).shuffle(100000))
    dataset = dataset.concatenate(dataset.map(random_jitter, num_parallel_calls=AUTOTUNE).shuffle(10000, reshuffle_each_iteration=True).repeat(repeats))
    dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE).shuffle(10000)
    return dataset


# + [markdown] papermill={"duration": 0.029543, "end_time": "2020-09-12T13:34:40.588890", "exception": false, "start_time": "2020-09-12T13:34:40.559347", "status": "completed"} tags=[]
# Then load the data and display the first images to see if it all worked out. Which of course it does, because it's taken directly from the tutorial.

# + papermill={"duration": 0.705509, "end_time": "2020-09-12T13:34:41.324127", "exception": false, "start_time": "2020-09-12T13:34:40.618618", "status": "completed"} tags=[]
monet_ds = load_dataset(MONET_FILENAMES, labeled=True, repeats=50).batch(100, drop_remainder=True)
photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True, repeats=2  ).batch(100, drop_remainder=True)


# + papermill={"duration": 9.225742, "end_time": "2020-09-12T13:34:50.579655", "exception": false, "start_time": "2020-09-12T13:34:41.353913", "status": "completed"} tags=[]
def view_image(ds, rows=2):
    image = next(iter(ds)) # extract 1 batch from the dataset
    image = image.numpy()

    fig = plt.figure(figsize=(22, rows * 5.05 ))
    for i in range(5 * rows):
        ax = fig.add_subplot(rows, 5, i+1, xticks=[], yticks=[])
        ax.imshow(image[i] / 2 + .5)

view_image(monet_ds)

# + papermill={"duration": 13.60154, "end_time": "2020-09-12T13:35:04.239599", "exception": false, "start_time": "2020-09-12T13:34:50.638059", "status": "completed"} tags=[]
view_image(photo_ds)

# + [markdown] papermill={"duration": 0.089974, "end_time": "2020-09-12T13:35:04.420385", "exception": false, "start_time": "2020-09-12T13:35:04.330411", "status": "completed"} tags=[]
# # Build the DCGAN
# ## Network Upsample and Downsample
#
# This one is the same from part1.
#
# The `downsample`, as the name suggests, reduces the 2D dimensions, the width and height, of the image by the stride. The stride is the length of the step the filter takes. Since the stride is 2, the filter is applied to every other pixel, hence reducing the weight and height by 2.
#
# We'll be using an instance normalization instead of batch normalization. As the instance normalization is not standard in the TensorFlow API, we'll use the layer from TensorFlow Add-ons.

# + papermill={"duration": 0.092368, "end_time": "2020-09-12T13:35:04.592135", "exception": false, "start_time": "2020-09-12T13:35:04.499767", "status": "completed"} tags=[]
OUTPUT_CHANNELS = 3
LATENT_DIM = 1024

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    result.add(layers.MaxPool2D())

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result


# + [markdown] papermill={"duration": 0.077976, "end_time": "2020-09-12T13:35:04.748830", "exception": false, "start_time": "2020-09-12T13:35:04.670854", "status": "completed"} tags=[]
# `Upsample` does the opposite of downsample and increases the dimensions of the of the image. `Conv2DTranspose` does basically the opposite of a `Conv2D` layer.

# + papermill={"duration": 0.092465, "end_time": "2020-09-12T13:35:04.920060", "exception": false, "start_time": "2020-09-12T13:35:04.827595", "status": "completed"} tags=[]
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.LeakyReLU())

    return result


# + [markdown] papermill={"duration": 0.079342, "end_time": "2020-09-12T13:35:05.077924", "exception": false, "start_time": "2020-09-12T13:35:04.998582", "status": "completed"} tags=[]
# ## Build Network
# The generator first downsamples the input image and then upsample while establishing long skip connections. Skip connections are a way to help bypass the vanishing gradient problem by concatenating the output of a layer to multiple layers instead of only one. Here we concatenate the output of the downsample layer to the upsample layer in a symmetrical fashion. Unets are pretty versatile and help out in our Generator to distill the input image to a lower dimension and then back to the full size at the target.
#
# ![](https://i.imgur.com/7GE9nY1.png)
# [Source](https://github.com/HarisIqbal88/PlotNeuralNet)

# + papermill={"duration": 0.089188, "end_time": "2020-09-12T13:35:05.246309", "exception": false, "start_time": "2020-09-12T13:35:05.157121", "status": "completed"} tags=[]
EPOCHS = 25

LR_G = 2e-4
LR_D = 2e-4
beta_1 = .5

real_label = .9
fake_label = 0


# + papermill={"duration": 0.100448, "end_time": "2020-09-12T13:35:05.425705", "exception": false, "start_time": "2020-09-12T13:35:05.325257", "status": "completed"} tags=[]
def CycleGenerator():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


# + [markdown] papermill={"duration": 0.078645, "end_time": "2020-09-12T13:35:05.583172", "exception": false, "start_time": "2020-09-12T13:35:05.504527", "status": "completed"} tags=[]
# The discriminator does not need a Unet, just a nice simple downsample to get a simple `fake` or `real` represented in numbers.

# + papermill={"duration": 0.096969, "end_time": "2020-09-12T13:35:05.759034", "exception": false, "start_time": "2020-09-12T13:35:05.662065", "status": "completed"} tags=[]
def CycleDiscriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last_conv = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    last_relu = layers.LeakyReLU(alpha=0.2)(last_conv)
    last_pool = layers.Flatten()(last_relu)
    last = layers.Dense(1, activation='sigmoid')(last_pool)

    return tf.keras.Model(inputs=inp, outputs=last)


# + papermill={"duration": 10.064908, "end_time": "2020-09-12T13:35:15.903037", "exception": false, "start_time": "2020-09-12T13:35:05.838129", "status": "completed"} tags=[]
with strategy.scope():
    monet_cycle_generator = CycleGenerator() # transforms photos to Monet-esque paintings
    photo_cycle_generator = CycleGenerator() # transforms Monet paintings to be more like photos

    monet_cycle_discriminator = CycleDiscriminator() # differentiates real Monet paintings and generated Monet paintings
    photo_cycle_discriminator = CycleDiscriminator() # differentiates real photos and generated photos


# + [markdown] papermill={"duration": 0.08043, "end_time": "2020-09-12T13:35:16.064048", "exception": false, "start_time": "2020-09-12T13:35:15.983618", "status": "completed"} tags=[]
# ## Build the CycleGAN model
#
# We will subclass a `tf.keras.Model` so that we can run `fit()` later to train our model. During the training step, the model transforms a photo to a Monet painting and then back to a photo. The difference between the original photo and the twice-transformed photo is the cycle-consistency loss. We want the original photo and the twice-transformed photo to be similar to one another.
#
# The way this works is by having one GAN for the forwards transformation and one GAN for the backwards transformation. So from image domain $X \rightarrow Y$ and backwards $Y \rightarrow X$. The resulting images are each evaluated by the standard discriminators of the GANs.
# ![](https://i.imgur.com/05Cjt6e.png)
#
# The losses are defined in the next section.

# + papermill={"duration": 0.118387, "end_time": "2020-09-12T13:35:16.263017", "exception": false, "start_time": "2020-09-12T13:35:16.144630", "status": "completed"} tags=[]
class CycleGan(keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
        real_label=.5
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        self.real_label = real_label
        
    def compile(
        self,
        m_gen_optimizer,
        p_gen_optimizer,
        m_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        
        batch_size = tf.shape(real_photo)[0]
        labels_real = tf.zeros((batch_size, 1)) + self.real_label
        labels_real += 0.05 * tf.random.uniform(tf.shape(labels_real))        
        
        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_real_monet, disc_fake_monet, labels_real)
            photo_gen_loss = self.gen_loss_fn(disc_real_photo, disc_fake_photo, labels_real)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet, labels_real)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo, labels_real)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        
        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }


# + [markdown] papermill={"duration": 0.080144, "end_time": "2020-09-12T13:35:16.422836", "exception": false, "start_time": "2020-09-12T13:35:16.342692", "status": "completed"} tags=[]
# ## Define loss functions
#
# The discriminator loss function below compares real images to a matrix of 1s and fake images to a matrix of 0s. The perfect discriminator will output all 1s for real images and all 0s for fake images. The discriminator loss outputs the average of the real and generated loss.
#
# The generator wants to fool the discriminator into thinking the generated image is real. The perfect generator will have the discriminator output only 1s. Thus, it compares the generated image to a matrix of 1s to find the loss.
#
# We could probably check if the Least Squares would also perform better for the Cycle GAN, so instead of just stupidly copying the starter, let's see if we can get this to improve the solution!

# + papermill={"duration": 0.093167, "end_time": "2020-09-12T13:35:16.596628", "exception": false, "start_time": "2020-09-12T13:35:16.503461", "status": "completed"} tags=[]
with strategy.scope():
    def discriminator_loss(predictions_real, predictions_gen, labels_real):
        return (tf.reduce_mean((predictions_gen  - tf.reduce_mean(predictions_real) + labels_real) ** 2) +
                tf.reduce_mean((predictions_real - tf.reduce_mean(predictions_gen)  - labels_real) ** 2))/2
    
    def generator_loss(predictions_real, predictions_gen, labels_real):
        return (tf.reduce_mean((predictions_real - tf.reduce_mean(predictions_gen)  + labels_real) ** 2) +
                tf.reduce_mean((predictions_gen  - tf.reduce_mean(predictions_real) - labels_real) ** 2)) / 2

# + [markdown] papermill={"duration": 0.079432, "end_time": "2020-09-12T13:35:16.755897", "exception": false, "start_time": "2020-09-12T13:35:16.676465", "status": "completed"} tags=[]
# ## More Loss Functions
# We want our original photo and the twice transformed photo to be similar to one another. Thus, we can calculate the cycle consistency loss be finding the average of their difference.

# + papermill={"duration": 0.089378, "end_time": "2020-09-12T13:35:16.924834", "exception": false, "start_time": "2020-09-12T13:35:16.835456", "status": "completed"} tags=[]
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1

# + [markdown] papermill={"duration": 0.079719, "end_time": "2020-09-12T13:35:17.085159", "exception": false, "start_time": "2020-09-12T13:35:17.005440", "status": "completed"} tags=[]
# The identity loss compares the image with its generator (i.e. photo with photo generator). If given a photo as input, we want it to generate the same image as the image was originally a photo. The identity loss compares the input with the output of the generator.

# + papermill={"duration": 0.090622, "end_time": "2020-09-12T13:35:17.256703", "exception": false, "start_time": "2020-09-12T13:35:17.166081", "status": "completed"} tags=[]
with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss

# + [markdown] papermill={"duration": 0.080388, "end_time": "2020-09-12T13:35:17.417517", "exception": false, "start_time": "2020-09-12T13:35:17.337129", "status": "completed"} tags=[]
# # Train the CycleGAN
#
# Let's compile our model. Since we used `tf.keras.Model` to build our CycleGAN, we can just ude the `fit` function to train our model. You know the drill already.

# + papermill={"duration": 0.09127, "end_time": "2020-09-12T13:35:17.589472", "exception": false, "start_time": "2020-09-12T13:35:17.498202", "status": "completed"} tags=[]
with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.5)

# + [markdown] papermill={"duration": 0.081278, "end_time": "2020-09-12T13:35:17.751555", "exception": false, "start_time": "2020-09-12T13:35:17.670277", "status": "completed"} tags=[]
# Double the optimizers double the fun!

# + papermill={"duration": 0.126906, "end_time": "2020-09-12T13:35:17.959990", "exception": false, "start_time": "2020-09-12T13:35:17.833084", "status": "completed"} tags=[]
with strategy.scope():
    cycle_gan_model = CycleGan(
        monet_cycle_generator, photo_cycle_generator, monet_cycle_discriminator, photo_cycle_discriminator, real_label=0.66
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

# + [markdown] papermill={"duration": 0.08207, "end_time": "2020-09-12T13:35:18.126232", "exception": false, "start_time": "2020-09-12T13:35:18.044162", "status": "completed"} tags=[]
# And finally we get to train!

# + _kg_hide-output=true papermill={"duration": 4324.047892, "end_time": "2020-09-12T14:47:22.259980", "exception": false, "start_time": "2020-09-12T13:35:18.212088", "status": "completed"} tags=[]
cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=EPOCHS
)

# + [markdown] papermill={"duration": 3.222844, "end_time": "2020-09-12T14:47:28.691739", "exception": false, "start_time": "2020-09-12T14:47:25.468895", "status": "completed"} tags=[]
# # Visualize our Monet-esque photos
# And now the CycleGAN with the augmented training data and LS Gan. Probably not that much different from before, but it's worth a try, right?

# + papermill={"duration": 24.933182, "end_time": "2020-09-12T14:47:56.882845", "exception": false, "start_time": "2020-09-12T14:47:31.949663", "status": "completed"} tags=[]
_, ax = plt.subplots(2, 5, figsize=(25, 5))
for i, img in enumerate(photo_ds.take(5)):
    prediction = monet_cycle_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[0, i].imshow(img)
    ax[1, i].imshow(prediction)
    ax[0, i].set_title("Input Photo")
    ax[1, i].set_title("Monet-esque")
    ax[0, i].axis("off")
    ax[1, i].axis("off")
plt.show()

# + [markdown] papermill={"duration": 3.325538, "end_time": "2020-09-12T14:48:03.444710", "exception": false, "start_time": "2020-09-12T14:48:00.119172", "status": "completed"} tags=[]
# # Create submission files
# We'll create the second submission file, as this is more of a tutorial so people can have a look at the outputs. Definitely make sure to play around with it!

# + papermill={"duration": 4.040691, "end_time": "2020-09-12T14:48:10.684780", "exception": false, "start_time": "2020-09-12T14:48:06.644089", "status": "completed"} tags=[]
import PIL
# ! mkdir ../images

# + papermill={"duration": 1561.99823, "end_time": "2020-09-12T15:14:16.010993", "exception": false, "start_time": "2020-09-12T14:48:14.012763", "status": "completed"} tags=[]
for i, img in enumerate(photo_ds.take(9999)):
    prediction = monet_cycle_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    im = PIL.Image.fromarray(prediction)
    im.save("../images/" + str(i) + ".jpg")

# + papermill={"duration": 3.591966, "end_time": "2020-09-12T15:14:22.793436", "exception": false, "start_time": "2020-09-12T15:14:19.201470", "status": "completed"} tags=[]
import shutil
shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")
