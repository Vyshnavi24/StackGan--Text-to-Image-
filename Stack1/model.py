import pickle
import random
import time

import numpy as np

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Dense, LeakyReLU, Concatenate, ReLU, Reshape, UpSampling2D, Conv2D, BatchNormalization, \
    Activation, Flatten, Lambda, concatenate
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib import pyplot as plt

embedDim = 128
GFDIM = 128
imsize = 64
s2, s4, s8, s16 = int(imsize / 2), int(imsize/ 4), int(imsize / 8), int(imsize / 16)
batchSize = 64
epochs = 100
zdim = 100

# Load training data: Images, embeddings are loaded from dataset.
trainPicklePath = '/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/Data/birds/train'
imageFileName = '/64images.pickle'
with open(trainPicklePath + imageFileName,'rb') as f:
    print('Loading images...')
    images = np.array(pickle.load(f))
    #print(images)

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
    embeddingIndex = random.randint(0, embeddings1.shape[0]-1)
    embed1 = embeddings1[embeddingIndex,:]
    embeddings.append(embed1)

embeddings = np.array(embeddings)

# Load testing data
testPicklePath = '/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/Data/birds/test'
imageFileName = '/64images.pickle'
with open(testPicklePath + imageFileName,'rb') as f:
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
    embeddingIndex = random.randint(0, embeddings1.shape[0]-1)
    embed1 = embeddings1[embeddingIndex,:]
    testembeddings.append(embed1)

testembeddings = np.array(testembeddings)


def KLLoss(yTrue, yPredict):
    mean = yPredict[:, :embedDim]
    logsigma = yPredict[:, embedDim:]
    return K.mean(- logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mean)))

"""
conditional augmentation for embedding
"""

textEmbed = Input(shape=(embeddings.shape[1],))
embedLayer = Dense(embedDim * 2)(textEmbed)
mu_sigma = LeakyReLU(alpha=0.2)(embedLayer)
mean = mu_sigma[:, :embedDim]
logsigma = mu_sigma[:, embedDim:]
sigma = K.exp(logsigma)
noise = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
chat = noise * sigma + mean

conditionalAugmentation = Model(inputs=[textEmbed], outputs=[mu_sigma])
conditionalAugmentation.compile(loss="binary_crossentropy", optimizer="adam")

# KL divergence
KLloss = K.mean(- logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mean)))


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
stage1Input = Input(shape=(100,))
inputLayer = Concatenate(axis=1)([Lambda(genChat)(mu_sigma), stage1Input])
stage1 = Dense(GFDIM * s16 * s16 * 8, use_bias=False)(inputLayer)
stage1 = ReLU()(stage1)

stage1 = Reshape((s16, s16, GFDIM * 8), input_shape=(GFDIM * s16 * s16 * 8,))(stage1)

stage1 = upsampleBlock(stage1, GFDIM * 4)
stage1 = upsampleBlock(stage1, GFDIM * 2)
stage1 = upsampleBlock(stage1, GFDIM)
stage1 = upsampleBlock(stage1, imsize)

stage1 = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(stage1)
stage1 = Activation(activation='tanh')(stage1)

generator = Model(inputs=[textEmbed, stage1Input], outputs=[stage1, mu_sigma])

genOptimizer = Adam(lr=genLearningRate, beta_1=0.5)
generator.compile(loss='mse',optimizer=genOptimizer)


def downsamplingBlock(stage1, units):
    stage1 = Conv2D(units, (4,4), padding='same', strides=2, use_bias=False)(stage1)
    stage1 = BatchNormalization()(stage1)
    stage1 = LeakyReLU(alpha=0.2)(stage1)
    return stage1


"""
building stage 1 discriminator
This block has two inputs: 64x64 image generated from above generator and text embedding.
So creating two input layers. 
"""
discLearningRate = 0.0002
# Input 1
stage1Input1 = Input(shape=(imsize, imsize, 3))
stage1Desc = Conv2D(imsize, (4,4), padding='same', strides=2, input_shape=(64, 64, 3), use_bias=False)(stage1Input1)
stage1Desc = LeakyReLU(alpha=0.2)(stage1Desc)

stage1Desc = downsamplingBlock(stage1Desc, GFDIM)
stage1Desc = downsamplingBlock(stage1Desc, GFDIM * 2)
stage1Desc = downsamplingBlock(stage1Desc, GFDIM * 4)

# Input 2
stage1Input2 = Input(shape=(4,4,128))
# check concatenate
concInput = concatenate([stage1Desc, stage1Input2])
descEmb = Conv2D(64 * 8, kernel_size=1,
                padding="same", strides=1)(concInput)
descEmb = BatchNormalization()(descEmb)
descEmb = LeakyReLU(alpha=0.2)(descEmb)
descEmb = Flatten()(descEmb)
descEmb = Dense(1)(descEmb)
descEmb = Activation('sigmoid')(descEmb)

discriminator = Model(inputs=[stage1Input1, stage1Input2], outputs=[descEmb])
discOptimizer = Adam(lr=discLearningRate, beta_1=0.5)
discriminator.compile(optimizer=discOptimizer, loss='binary_crossentropy')


"""
Compressed embedding
"""
textEmbedComp = Input(shape=(embeddings.shape[1],))
embedLayerComp = Dense(embedDim)(textEmbedComp)
embedComp = LeakyReLU(alpha=0.2)(embedLayerComp)
compModel = Model(inputs=[textEmbedComp], outputs=[embedComp])
compModel.compile(loss='binary_crossentropy', optimizer='adam')

"""
Adversary model
"""
CALayer = Input(shape=(1024,))
genLayer = Input(shape=(100,))
discLayer = Input(shape=(4, 4, 128))

xImage, mean_sigma = generator([CALayer, genLayer])
discriminator.trainable = False
disc = discriminator([xImage, discLayer])

GAN = Model(inputs=[CALayer, genLayer, discLayer], outputs=[disc, mean_sigma])
GAN.compile(loss=['binary_crossentropy', KLLoss], loss_weights=[1,2], optimizer=genOptimizer, metrics=None)

board = TensorBoard(log_dir='/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/logs'.format(time.time()))
board.set_model(generator)
board.set_model(discriminator)
board.set_model(conditionalAugmentation)
board.set_model(compModel)

realLabels = np.ones((batchSize,1), dtype=float) * 0.9
fakeLabels = np.zeros((batchSize,1), dtype=float) * 0.1


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
    print('=====================================')
    print('Epoch:', epoch)
    generatorLoss = []
    discriminatorLoss = []
    numberOfBatches = int(images.shape[0]/batchSize)

    for index in range(numberOfBatches):
        print("Batch Number:", index+1)

        z = np.random.normal(0, 1, size=(batchSize, zdim))
        trainBatch = images[index * batchSize: (index + 1) * batchSize, :, :, :]
        #print(trainBatch)
        embeddingBatch = embeddings[index * batchSize: (index + 1) * batchSize]
        trainBatch = (trainBatch - 127.5)/127.5

        fakeImages, _ = generator.predict([embeddingBatch, z], verbose=3)

        compEmbeddings = compModel.predict_on_batch(embeddingBatch)
        compEmbeddings = np.reshape(compEmbeddings, (-1, 1, 1, 128))
        compEmbeddings = np.tile(compEmbeddings, (1, 4, 4, 1))
        realLoss = discriminator.train_on_batch([trainBatch, compEmbeddings], np.reshape(realLabels, (batchSize, 1)))
        fakeLoss = discriminator.train_on_batch([fakeImages, compEmbeddings], np.reshape(fakeLabels, (batchSize, 1)))
        wrongLoss = discriminator.train_on_batch([trainBatch[:(batchSize - 1)], compEmbeddings[1:]], np.reshape(fakeLabels[1:], (batchSize-1, 1)))

        discLoss = 0.5 * np.add(realLoss, 0.5 * np.add(wrongLoss, fakeLoss))
        print("discLoss:{}".format(discLoss))
        genLoss = GAN.train_on_batch([embeddingBatch, z, compEmbeddings],
                                                  [K.ones((batchSize, 1)) * 0.9, K.ones((batchSize, 256)) * 0.9])
        print("genLoss:{}".format(genLoss))

        discriminatorLoss.append(discLoss)
        generatorLoss.append(genLoss)

    write_log(board, 'discriminator_loss', np.mean(discriminatorLoss), epoch)
    write_log(board, 'generator_loss', np.mean(generatorLoss[0]), epoch)

    if epoch % 10 == 0:
        z2 = np.random.normal(0, 1, size=(batchSize, 100))
        numberOfTestBatches = int(testimages.shape[0]/batchSize)
        for index1 in range(numberOfTestBatches):
            testembeddingBatch = testembeddings[index1 * batchSize: (index1 + 1) * batchSize]
            fakeImagesTest,_ = generator.predict_on_batch([testembeddingBatch, z2])

            for i, img in enumerate(fakeImagesTest[:10]):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("Image")

                plt.savefig("/Users/shrinidhi/study/StudyMaterial/sem2/machine_learning/project/StackGAN/results/gen_{}_{}_{}.png".format(epoch,index1, i))
                plt.close()

    generator.save_weights("stage1_gen.h5")
    discriminator.save_weights("stage1_dis.h5")


