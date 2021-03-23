import pickle
import random
import time

import numpy as np

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Dense, LeakyReLU, Concatenate, ReLU, Reshape, UpSampling2D, Conv2D, BatchNormalization, \
    Activation, Flatten, Lambda, concatenate, add, ZeroPadding2D
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib import pyplot as plt

embedDim = 128
GFDIM = 128
imsize = 256
s2, s4, s8, s16 = int(imsize / 2), int(imsize / 4), int(imsize / 8), int(imsize / 16)
batchSize = 4
epochs = 100
zdim = 100

# Load training data: Images, embeddings are loaded from dataset.
trainPicklePath = '/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/Data/birds/train'
imageFileName = '/256images.pickle'
with open(trainPicklePath + imageFileName, 'rb') as f:
    print('Loading images...')
    images = np.array(pickle.load(f))
    print(images[0].shape)

embedding_filename = '/char-CNN-RNN-embeddings.pickle'
with open(trainPicklePath + embedding_filename, 'rb') as f:
    print('Loading embeddings...')
    embedding = np.array(pickle.load(f, encoding="bytes"))

fileNames_file = '/filenames.pickle'
with open(trainPicklePath + fileNames_file, 'rb') as f:
    fileNames = pickle.load(f)

class_info = '/class_info.pickle'
with open(trainPicklePath + class_info, 'rb') as f:
    class_id = pickle.load(f, encoding="bytes")

embeddings = []

for i, file in enumerate(fileNames):
    embeddings1 = embedding[i, :, :]
    embeddingIndex = random.randint(0, embeddings1.shape[0] - 1)
    embed1 = embeddings1[embeddingIndex, :]
    embeddings.append(embed1)

embeddings = np.array(embeddings)

# Load testing data
testPicklePath = '/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/Data/birds/test'
imageFileName = '/256images.pickle'
with open(testPicklePath + imageFileName, 'rb') as f:
    print('Loading images...')
    testimages = np.array(pickle.load(f))

testembedding_filename = '/char-CNN-RNN-embeddings.pickle'
with open(testPicklePath + testembedding_filename, 'rb') as f:
    print('Loading embeddings...')
    testembedding = np.array(pickle.load(f, encoding="bytes"))

testfileNames_file = '/filenames.pickle'
with open(testPicklePath + testfileNames_file, 'rb') as f:
    testfileNames = pickle.load(f)

testclass_info = '/class_info.pickle'
with open(testPicklePath + testclass_info, 'rb') as f:
    testclass_id = pickle.load(f, encoding="bytes")

testembeddings = []

for i, file in enumerate(fileNames):
    embeddings1 = embedding[i, :, :]
    embeddingIndex = random.randint(0, embeddings1.shape[0] - 1)
    embed1 = embeddings1[embeddingIndex, :]
    testembeddings.append(embed1)

testembeddings = np.array(testembeddings)


def KLLoss(yTrue, yPredict):
    mean = yPredict[:, :embedDim]
    logsigma = yPredict[:, embedDim:]
    return K.mean(- logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mean)))


def downsamplingBlock(stage1, units):
    stage1 = Conv2D(units, (4, 4), padding='same', strides=2, use_bias=False)(stage1)
    stage1 = BatchNormalization()(stage1)
    stage1 = LeakyReLU(alpha=0.2)(stage1)
    return stage1


"""
building stage 2 discriminator
This block has two inputs: 64x64 image generated from above generator and text embedding.
So creating two input layers. 
"""
discLearningRate = 0.0002
# Input 1
stage2Input1 = Input(shape=(imsize, imsize, 3))
stage2Desc = Conv2D(64, (4, 4), padding='same', strides=2, input_shape=(256, 256, 3), use_bias=False)(stage2Input1)
stage2Desc = LeakyReLU(alpha=0.2)(stage2Desc)

stage2Desc = downsamplingBlock(stage2Desc, GFDIM)
stage2Desc = downsamplingBlock(stage2Desc, GFDIM * 2)
stage2Desc = downsamplingBlock(stage2Desc, GFDIM * 4)
stage2Desc = downsamplingBlock(stage2Desc, GFDIM * 8)
stage2Desc = downsamplingBlock(stage2Desc, GFDIM * 16)

stage2Desc = Conv2D(GFDIM * 8, (1, 1), padding='same', strides=1, use_bias=False)(stage2Desc)
stage2Desc = BatchNormalization()(stage2Desc)
stage2Desc = LeakyReLU(alpha=0.2)(stage2Desc)

stage2Desc = Conv2D(GFDIM * 4, (1, 1), padding='same', strides=1, use_bias=False)(stage2Desc)
stage2Desc = BatchNormalization()(stage2Desc)

stage2Desc2 = Conv2D(GFDIM, (1, 1), padding='same', strides=1, use_bias=False)(stage2Desc)
stage2Desc2 = BatchNormalization()(stage2Desc2)
stage2Desc2 = LeakyReLU(alpha=0.2)(stage2Desc2)

stage2Desc2 = Conv2D(GFDIM, (3, 3), padding='same', strides=1, use_bias=False)(stage2Desc2)
stage2Desc2 = BatchNormalization()(stage2Desc2)
stage2Desc2 = LeakyReLU(alpha=0.2)(stage2Desc2)

stage2Desc2 = Conv2D(GFDIM * 4, (3, 3), padding='same', strides=1, use_bias=False)(stage2Desc2)
stage2Desc2 = BatchNormalization()(stage2Desc2)

add_desc = add([stage2Desc, stage2Desc2])
add_desc = LeakyReLU(alpha=0.2)(add_desc)

# Input 2
stage2Input2 = Input(shape=(4, 4, 128))
# check concatenate
concInput2 = concatenate([add_desc, stage2Input2])
descEmb2 = Conv2D(64 * 8, kernel_size=1,
                  padding="same", strides=1)(concInput2)
descEmb2 = BatchNormalization()(descEmb2)
descEmb2 = LeakyReLU(alpha=0.2)(descEmb2)
descEmb2 = Flatten()(descEmb2)
descEmb2 = Dense(1)(descEmb2)
descEmb2 = Activation('sigmoid')(descEmb2)

discriminator2 = Model(inputs=[stage2Input1, stage2Input2], outputs=[descEmb2])
discOptimizer2 = Adam(lr=discLearningRate, beta_1=0.5)
discriminator2.compile(optimizer=discOptimizer2, loss='binary_crossentropy')

"""
Stage 1 generator. same as model in stack 1
"""


def upsampleBlock(stage1, units):
    stage1 = UpSampling2D(size=(2, 2))(stage1)
    stage1 = Conv2D(units, kernel_size=3, padding="same", strides=1, use_bias=False)(stage1)
    stage1 = BatchNormalization()(stage1)
    stage1 = ReLU()(stage1)
    return stage1


# building Stage 1 Generator
def genChat(mu_sigma):
    mean = mu_sigma[:, :embedDim]
    logsigma = mu_sigma[:, embedDim:]
    sigma = K.exp(logsigma)
    noise = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    chat = noise * sigma + mean
    return chat


genLearningRate = 0.0002

textEmbed = Input(shape=(embeddings.shape[1],))
embedLayer = Dense(embedDim * 2)(textEmbed)
mu_sigma = LeakyReLU(alpha=0.2)(embedLayer)

stage1Input = Input(shape=(100,))

inputLayer = Concatenate(axis=1)([Lambda(genChat)(mu_sigma), stage1Input])
stage1 = Dense(128 * 4 * 4 * 8, use_bias=False)(inputLayer)
stage1 = ReLU()(stage1)

stage1 = Reshape((4, 4, 128 * 8), input_shape=(128 * 4 * 4 * 8,))(stage1)

stage1 = upsampleBlock(stage1, GFDIM * 4)
stage1 = upsampleBlock(stage1, GFDIM * 2)
stage1 = upsampleBlock(stage1, GFDIM)
stage1 = upsampleBlock(stage1, 64)

stage1 = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(stage1)
stage1 = Activation(activation='tanh')(stage1)

generator1 = Model(inputs=[textEmbed, stage1Input], outputs=[stage1, mu_sigma])

genOptimizer1 = Adam(lr=genLearningRate, beta_1=0.5)
generator1.compile(loss='binary_crossentropy', optimizer=genOptimizer1)

generator1.load_weights(
    "/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/Stack1/stage1_gen.h5")

"""
Stage 2 generator
"""

"""
Conditional augmenting
"""

textEmbed2 = Input(shape=(embeddings.shape[1],))
LrInput = Input(shape=(64, 64, 3))

embedLayer2 = Dense(embedDim * 2)(textEmbed2)
mu_sigma = LeakyReLU(alpha=0.2)(embedLayer2)
chat2 = Lambda(genChat)(mu_sigma)

"""
Image Generation
"""


def joint_block(inputs):
    c = inputs[0]
    x = inputs[1]

    c = K.expand_dims(c, axis=1)
    c = K.expand_dims(c, axis=1)
    c = K.tile(c, [1, 16, 16, 1])
    return K.concatenate([c, x], axis=3)


def residualBlock(input):
    """
       Residual block in the generator network
       """
    x = Conv2D(128 * 4, kernel_size=(3, 3), padding='same', strides=1)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128 * 4, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, input])
    x = ReLU()(x)

    return x


stage2Gen = ZeroPadding2D(padding=(1, 1))(LrInput)
stage2Gen = Conv2D(GFDIM, kernel_size=(3, 3), strides=1, use_bias=False)(stage2Gen)
stage2Gen = ReLU()(stage2Gen)

stage2Gen = ZeroPadding2D(padding=(1, 1))(stage2Gen)
stage2Gen = Conv2D(GFDIM * 2, kernel_size=(4, 4), strides=2, use_bias=False)(stage2Gen)
stage2Gen = ReLU()(stage2Gen)

stage2Gen = ZeroPadding2D(padding=(1, 1))(stage2Gen)
stage2Gen = Conv2D(GFDIM * 4, kernel_size=(4, 4), strides=2, use_bias=False)(stage2Gen)
stage2Gen = ReLU()(stage2Gen)

concInput2gen = Lambda(joint_block)([chat2, stage2Gen])

stage2Gen = ZeroPadding2D(padding=(1, 1))(concInput2gen)
stage2Gen = Conv2D(GFDIM * 4, kernel_size=(3, 3), strides=1, use_bias=False)(stage2Gen)
stage2Gen = BatchNormalization()(stage2Gen)
stage2Gen = ReLU()(stage2Gen)

# Residual

stage2Gen = residualBlock(stage2Gen)
stage2Gen = residualBlock(stage2Gen)
stage2Gen = residualBlock(stage2Gen)
stage2Gen = residualBlock(stage2Gen)

# upsampling blocks

stage2Gen = upsampleBlock(stage2Gen, GFDIM * 4)
stage2Gen = upsampleBlock(stage2Gen, GFDIM * 2)
stage2Gen = upsampleBlock(stage2Gen, GFDIM)
stage2Gen = upsampleBlock(stage2Gen, 64)
stage2Gen = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(stage2Gen)
stage2Gen = Activation('tanh')(stage2Gen)

generator2 = Model(inputs=[textEmbed2, LrInput], outputs=[stage2Gen, mu_sigma])
genOptimizer1 = Adam(lr=genLearningRate, beta_1=0.5)
generator2.compile(loss='binary_crossentropy', optimizer=genOptimizer1)

"""
Embedding compressor
"""
textEmbedComp = Input(shape=(embeddings.shape[1],))
embedLayerComp = Dense(embedDim)(textEmbedComp)
embedComp = LeakyReLU(alpha=0.2)(embedLayerComp)
compModel = Model(inputs=[textEmbedComp], outputs=[embedComp])
compModel.compile(loss='binary_crossentropy', optimizer='adam')

"""
Adverserial model
"""
CALayer = Input(shape=(1024,))
genLayer = Input(shape=(100,))
discLayer = Input(shape=(4, 4, 128))

generator1.trainable = False
discriminator2.trainable = False

lr_images, mean_logsigma1 = generator1([CALayer, genLayer])
hr_images, mean_logsigma2 = generator2([CALayer, lr_images])
disc = discriminator2([hr_images, discLayer])

GAN = Model(inputs=[CALayer, genLayer, discLayer], outputs=[disc, mean_logsigma2])
GAN.compile(loss=['binary_crossentropy', KLLoss], loss_weights=[1.0, 2.0],
            optimizer=genOptimizer1, metrics=None)

tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
tensorboard.set_model(generator2)
tensorboard.set_model(discriminator2)

realLabels = np.ones((batchSize, 1), dtype=float) * 0.9
fakeLabels = np.zeros((batchSize, 1), dtype=float) * 0.1


def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


for epoch in range(epochs):
    print("========================================")
    print("Epoch is:", epoch)

    gen_losses = []
    dis_losses = []
    # Load data and train model
    numberOfBatches = int(images.shape[0] / batchSize)
    print("Number of batches:{}".format(numberOfBatches))
    for index in range(numberOfBatches):
        print("Batch:{}".format(index + 1))

        # Create a noise vector
        z_noise = np.random.normal(0, 1, size=(batchSize, zdim))
        X_hr_train_batch = images[index * batchSize:(index + 1) * batchSize, :, :, :]
        embedding_batch = embeddings[index * batchSize:(index + 1) * batchSize]
        X_hr_train_batch = (X_hr_train_batch - 127.5) / 127.5

        # Generate fake images
        lr_fake_images, _ = generator1.predict([embedding_batch, z_noise], verbose=3)
        hr_fake_images, _ = generator2.predict([embedding_batch, lr_fake_images], verbose=3)

        """
        4. Generate compressed embeddings
        """
        compressed_embedding = compModel.predict_on_batch(embedding_batch)
        compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, 128))
        compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

        """
        5. Train the discriminator model
        """
        dis_loss_real = discriminator2.train_on_batch([X_hr_train_batch, compressed_embedding],
                                                      np.reshape(realLabels, (batchSize, 1)))
        dis_loss_fake = discriminator2.train_on_batch([hr_fake_images, compressed_embedding],
                                                      np.reshape(fakeLabels, (batchSize, 1)))
        dis_loss_wrong = discriminator2.train_on_batch([X_hr_train_batch[:(batchSize - 1)], compressed_embedding[1:]],
                                                       np.reshape(fakeLabels[1:], (batchSize - 1, 1)))
        d_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))
        print("d_loss:{}".format(d_loss))

        """
        Train the adversarial model
        """
        g_loss = GAN.train_on_batch([embedding_batch, z_noise, compressed_embedding],
                                    [K.ones((batchSize, 1)) * 0.9, K.ones((batchSize, 256)) * 0.9])

        print("g_loss:{}".format(g_loss))

        dis_losses.append(d_loss)
        gen_losses.append(g_loss)

    """
    Save losses to Tensorboard after each epoch
    """
    write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
    write_log(tensorboard, 'generator_loss', np.mean(gen_losses)[0], epoch)

    # Generate and save images after every 2nd epoch
    if epoch % 2 == 0:
        z_noise2 = np.random.normal(0, 1, size=(batchSize, zdim))
        embedding_batch = testembeddings[0:batchSize]

        lr_fake_images, _ = generator1.predict([embedding_batch, z_noise2], verbose=3)
        hr_fake_images, _ = generator2.predict([embedding_batch, lr_fake_images], verbose=3)

        # Save images
        for i, img in enumerate(hr_fake_images[:10]):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("Image")

            plt.savefig(
                "/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/results/gen_{}_{}.png".format(
                    epoch, i))
            plt.close()

    # Saving the models
    generator2.save_weights("stage2_gen.h5")
    discriminator2.save_weights("stage2_dis.h5")
