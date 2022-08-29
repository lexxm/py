#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization, MaxPooling2D, LSTM, Activation, Lambda, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D, Conv3DTranspose
from keras.layers.merge import add, concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import random
import tensorflow as tf
import itertools
import math
from keras import backend as K

alphabet = u'0123456789'
np.random.seed(1337)
(x_train_list, y_train_list, x_test_list, y_test_list ) = ( [], [], [], [] )

alp_size = len(alphabet)
input_length_gl = 12
input_length_lcl = 6
size_emb = 100

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
def func1( y_true, y_pred ):
    return y_pred
    
def text_to_labels(text ):
    ret = []
    for char in text:
        if alphabet.find(char) == -1:
            print( "------- ", char, " ---------")
    for char in text:
        ret.append(alphabet.find(char))
    return ret

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def build_generator2( latent_size ):
    text_data_layer = Input(name='the_input_text_data_6', shape=[6], dtype='int32' )
    latent = Input(shape=[latent_size], name='the_input_latent', dtype='float32' )
    
    print( "alp_size:", alp_size)
    cls = Embedding( alp_size, size_emb, embeddings_initializer='glorot_normal')(text_data_layer)
    cls = Reshape( target_shape=[ latent_size ] )( cls )
    
    input_data = Multiply()( [cls, latent] )
    
    input_data = Dropout(0.1)(input_data)
    inner = Dense( 12*3*128, activation='relu' )( input_data )
    
    inner = Reshape( target_shape=(12, 3, 128) )( inner )

    inner = Conv2DTranspose(128, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal')( inner )
    inner = BatchNormalization()(inner)

    inner = Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal')( inner )
    inner = BatchNormalization()(inner)
    
    inner = Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal')( inner )
    inner = BatchNormalization()(inner)

    out = Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh', kernel_initializer='glorot_normal')( inner )

    gen_model = Model([latent, text_data_layer], out )
    gen_model.summary()
    return gen_model

def build_discriminator( ):
    labels = Input(name='the_input_text_data_20', shape=[input_length_gl], dtype='int32' )
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    input_shape = ( 192, 48, 1 )
    conv_filters = 16
    kernel_size = ( 3, 3 )
    rnn_size = 32
    pool_size = 2
    act = 'relu'
    
    input_data = Input(name='the_input_img', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same', kernel_initializer='he_normal', name='conv1')(input_data)
    inner = LeakyReLU(0.2)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    inner = Conv2D(conv_filters, kernel_size, padding='same', kernel_initializer='he_normal', name='conv2')(inner)
    inner = LeakyReLU(0.2)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    inner = Conv2D(conv_filters, kernel_size, padding='same', kernel_initializer='he_normal', name='conv3')(inner) 
    inner = LeakyReLU(0.2)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(inner)

    inner = Conv2D(conv_filters, kernel_size, padding='same', kernel_initializer='he_normal', name='conv4')(inner) 
    inner = LeakyReLU(0.2)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max4')(inner)

    ( img_w, img_h ) = ( input_shape[0], input_shape[1] )
    conv_to_rnn_dims = (img_w // (pool_size ** 4), (img_h // (pool_size ** 4)) * conv_filters )
    innerR = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    gru_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(innerR)
    gru_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm_b')(innerR)
    lstm_out = concatenate([gru_1, gru_1b])
    #print( "====" "=========")
    out_size = len(alphabet) + 1
    outD = Dense( out_size, kernel_initializer='he_normal', name='dense2')(lstm_out)
                 
    y_pred = Activation('softmax', name='softmax')(outD)

    fl2 = Flatten()(inner)
    fake = Dense(1, activation='sigmoid', name='generation')(fl2)
    loss_out = Lambda( ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=(loss_out,fake))
    
    print("discriminator summary")
    model.summary()
        
    return model
    
def loadImages( pathTxt, imagesDir ):
    #(x_train_list, y_train_list, x_test_list, y_test_list ) = ( [], [], [], [] )
    global x_train_list
    global y_train_list
    global x_test_list
    global y_test_list
    
    f=open(pathTxt)
    maxTextSize = 0
    count_stroke = 0
    wrong_text_len = 0
    wrong_alp = 0
    loaded_img = 0
    for line in f:
        lineSplit = line.strip().split(' ')
        fileNameSplit = lineSplit[0]
        gtText = lineSplit[1]

        count_stroke += 1
        rand_value = random.randint( 1, 100 )        

        isBad = False
        for char in gtText:
            if alphabet.find(char) == -1:
                isBad = True
        if len(gtText) > 12 or len(gtText) < 2:
            wrong_text_len += 1
            continue
        if isBad:
            wrong_alp += 1
            continue
        str = imagesDir + "/" + fileNameSplit
        if len(gtText) > 6:
            print( count_stroke, gtText )

        im = Image.open( str ).convert('L')
        #im = im.resize( [160,40] )
        im = im.resize( [192,48] )
        image_data = np.asarray( im ).T

        if rand_value > 15:
            x_train_list.append( image_data )
            y_train_list.append( gtText )
        else:
            x_test_list.append( image_data )
            y_test_list.append( gtText )

        if len(gtText) > maxTextSize:
            maxTextSize = len(gtText)
        loaded_img += 1
    new_ind = 0

    while len(x_train_list) % 32 != 0:
        x_train_list.append( x_train_list[new_ind] )
        y_train_list.append( y_train_list[new_ind] )
        new_ind += 1
    #return ( x_train_list, y_train_list, x_text_list, y_text_list)

def createExtraArrays( label_batch_text, input_length_size ):
    batch_size_loc = len( label_batch_text )
    label_batch = np.ones([batch_size_loc, input_length_size])
    input_length = np.zeros([batch_size_loc, 1])
    label_length = np.zeros([batch_size_loc, 1])

    for i in range( batch_size_loc ):
        text = label_batch_text[i]
        label_length[i] = len(text)
        label_batch[i, :len(text)] = text_to_labels( text )
        input_length[i] = input_length_size-2

    inputs = { 'the_input_text_data': label_batch,
          'input_length': input_length,
          'label_length': label_length,
          'text' : label_batch_text }
    return inputs

def generateText( batch_size_l, input_length_size ):
    genText = []
    index_gen = np.random.randint(0, alp_size, ( batch_size_l, 6 ) )
    for i in range( batch_size_l ):
        text = str()
        index_gen[i][0] = 1
        for j in range(6):
            jj = index_gen[i][j]
            text = text + alphabet[jj]
        genText.append( text )
    gen_inputs_gen = createExtraArrays( genText, 6 )
    gen_inputs_comp = createExtraArrays( genText, input_length_size )
    return ( gen_inputs_gen, gen_inputs_comp )
    
def recognize( model, X_data3 ):
    t1 = tf.convert_to_tensor( model.layers[0].input )
    t_dense = tf.convert_to_tensor( model.layers[18].output )#13
    test_func = K.function([t1], [t_dense])

    out = test_func( [X_data3] )

    out = np.asarray( out )
    print( out.shape, X_data3.shape )
    out = out.reshape( out.shape[1], out.shape[2], out.shape[3] )
    ret = []
    ret_array = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        ret_array.append( out_best )
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ( ret_array, ret )
    

if __name__ == '__main__':
    loadImages( "/u01/share/indexes_all/WordNew.txt", "/u01/share/indexes_all" )
    print( "load 1 compleate" )
    
    x_train = ( np.asarray( x_train_list, dtype = 'float32' ) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.asarray( y_train_list )
    x_test = ( np.asarray( x_test_list, dtype = 'float32' ) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.asarray( y_test_list )
    
    print( x_train.shape )

    # batch and latent size taken from the paper
    epochs = 100
    batch_size = 32
    latent_size = 6*size_emb

    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    
    labels_6 = Input(name='the_input_text_data_6', shape=[6], dtype='int32' )
    labels_20 = Input(name='the_input_text_data_20', shape=[input_length_gl], dtype='int32' )
    latent = Input(shape=[latent_size], name='the_input_latent', dtype='float32' )
    input_length_t = Input(name='input_length', shape=[1], dtype='int32')
    label_length_t = Input(name='label_length', shape=[1], dtype='int32')
    
    discriminator = build_discriminator()
    
    discriminator.compile(
        optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        loss=[ func1, 'binary_crossentropy' ] )
    
    # build the generator
    print('Generator model:')
    generator = build_generator2( latent_size )
    generator.summary()

    fake = generator([latent, labels_6])

    discriminator.trainable = False
    
    loss_out, fake_flag = discriminator( [fake, labels_20, input_length_t, label_length_t] )
    combined = Model( [latent, labels_6, labels_20, input_length_t, label_length_t], [loss_out, fake_flag] )

    print('Combined model:')
    combined.compile(
        optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),
        #loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        loss=[ func1, 'binary_crossentropy' ]
    )
    print("combined summary")
    combined.summary()
    
    num_train, num_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(np.ceil(x_train.shape[0] / float(batch_size)))
        progress_bar = Progbar(target=num_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # get a batch of real images
            # загружаем хорошие изображения
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            #print( label_batch )
            inputs_gen = createExtraArrays( label_batch, 6 )# да это одни и теже данные, просто для разных размеров - один нужен для входа на генератор, другой для выхода дискриминатора
            inputs_disc = createExtraArrays( label_batch, input_length_gl )

            noise = np.random.uniform(-1, 1, (len(image_batch), latent_size ))

            ( gen_inputs_gen, gen_inputs_disc ) = generateText( batch_size, input_length_gl )
            generated_labels_gen = gen_inputs_gen['the_input_text_data']
            generated_labels_disc = gen_inputs_disc['the_input_text_data']

            generated_images = generator.predict( [noise, generated_labels_gen], verbose=0)
            x = np.concatenate((image_batch, generated_images))
            
            x2_6 = np.concatenate((inputs_gen['the_input_text_data'], gen_inputs_gen['the_input_text_data']))
            x2_20 = np.concatenate((inputs_disc['the_input_text_data'], gen_inputs_disc['the_input_text_data']))
            x3 = np.concatenate((inputs_disc['input_length'], gen_inputs_disc['input_length']))
            x4 = np.concatenate((inputs_disc['label_length'], gen_inputs_disc['label_length']))

            soft_zero, soft_one = 0, 0.95
            y = np.array( [soft_one] * len(image_batch) + [soft_zero] * len(image_batch))

            disc_sample_weight = [np.concatenate((np.ones(len(image_batch)) * 2, np.zeros(len(image_batch)))), np.ones(2 * len(image_batch))]
            outputs =  [ np.zeros(2*batch_size), y ]
            inputs = {'the_input_img': x, 'the_input_text_data_20': x2_20, 'input_length' : x3, 'label_length': x4 }
            tobr = discriminator.train_on_batch( inputs, outputs, sample_weight=disc_sample_weight )#
            epoch_disc_loss.append( tobr )
            # тренируем 1 эпоху (только дискриминатор)
            
            noise = np.random.uniform(-1, 1, (2 * len(image_batch), latent_size ))
            ( gen_inputs_gen, gen_inputs_disc ) = generateText( 2*batch_size, input_length_gl )
            gen_inputs = gen_inputs_disc
            gen_inputs['the_input_text_data_6'] = gen_inputs_gen['the_input_text_data']
            gen_inputs['the_input_text_data_20'] = gen_inputs_disc['the_input_text_data']
            trick = np.ones(2 * len(image_batch)) * soft_one

            # обучаем уже комбинированную связку
            gen_inputs['the_input_latent'] = noise
            gen_outputs = [ np.zeros(2*batch_size), trick ]
            epoch_gen_loss.append(combined.train_on_batch(
                gen_inputs,
                gen_outputs ))
            
            progress_bar.update(index + 1)

        print('Testing for epoch {}:'.format(epoch))
        
        noise = np.random.uniform(-1, 1, (num_test, latent_size))
        
        ( gen_inputs_gen, gen_inputs_disc ) = generateText( num_test, input_length_gl )
        generated_labels_gen = gen_inputs_gen['the_input_text_data']
        
        # создаем изображения для тестовk
        generated_images = generator.predict( [noise, generated_labels_gen], verbose=False)
        
        test_labels_gen = createExtraArrays( y_test, 6 )
        test_labels_disc = createExtraArrays( y_test, input_length_gl )
        test_labels_disc['the_input_text_data_20'] = test_labels_disc['the_input_text_data']

        # тестируем 
        x = np.concatenate((x_test, generated_images))
        x2_6 = np.concatenate((test_labels_gen['the_input_text_data'], gen_inputs_gen['the_input_text_data']))
        x2_20 = np.concatenate((test_labels_disc['the_input_text_data'], gen_inputs_disc['the_input_text_data']))
        x3 = np.concatenate((test_labels_disc['input_length'], gen_inputs_disc['input_length']))
        x4 = np.concatenate((test_labels_disc['label_length'], gen_inputs_disc['label_length']))
        
        y = np.array([1] * num_test + [0] * num_test)
        # подаем на вход картинки( сгенерированные и обычные ), на выходе ответы - считаем score
        outputs =  [ np.zeros(2*num_test), y ]
        inputs = {'the_input_img': x, 'the_input_text_data_20': x2_20, 'input_length' : x3, 'label_length': x4 }
        
        discriminator_test_loss = discriminator.evaluate( inputs, outputs, verbose=False )
        
        train_weight = np.mean(np.array(epoch_disc_loss), axis=0)
        print( "train disc average:", train_weight )
        print( "dics test: ", discriminator_test_loss )
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        #---------------------------------
        train_weight = np.mean(np.array(discriminator_test_loss))
        print( "dicriminator test:", train_weight )
        train_weight = np.mean(np.array( epoch_disc_loss ))
        print( "epoch_disc_loss:", train_weight )
        
        list1 = []
        for i in range(10):
            list1.append( x_test[i] )

        ( ret_list, str_list ) = recognize( discriminator, np.asarray( list1 ) )
        for i in range(10):
            print( str_list[i], "->", y_test[i] )

        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size ))
        ( gen_inputs_gen, gen_inputs_disc ) = generateText( 2*num_test, input_length_gl )
        
        test_inputs = gen_inputs_disc
        test_inputs['the_input_latent'] = noise
        test_inputs['the_input_text_data_6'] = gen_inputs_gen['the_input_text_data']
        test_inputs['the_input_text_data_20'] = gen_inputs_disc['the_input_text_data']

        trick = np.ones(2 * num_test)

        # теперь обрабатываем комбинированныую модель
        generator_test_loss = combined.evaluate(
            test_inputs,
            [np.zeros(2*num_test), trick ], verbose=False)
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # записываем лоссы в таблицу
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)
        
        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))
        
        # сохранение весов для кажой эпохи
        #generator.save_weights( 'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        #discriminator.save_weights( 'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)
        
        num_rows = 2
        
        for i in range(10):
            noise = np.random.uniform(-1, 1, (1, latent_size ))
            ( gen_inputs_gen, gen_inputs_disc ) = generateText( 1, input_length_gl )
            generated_labels = gen_inputs_gen['the_input_text_data']
            generated_images = generator.predict( [noise, generated_labels], verbose=0)
            text = gen_inputs_gen['text'][0]
            
            img_arr = generated_images[0].reshape( 192, 48 )
            img_arr = img_arr.T * 127.5 + 127.5
            Image.fromarray( np.uint8( img_arr) ).save( "plot12/%02d_%d_generated_%s.png" % (epoch, i, text) )
