import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from keras.layers import merge


def loglike_skew_cost(mu, sigma, skewness, y):
    dist = tfp.distributions.SinhArcsinh(loc=mu, scale=sigma, skewness=skewness)
    return tf.reduce_mean(-dist.log_prob(y))

def loglike_cost(mu, sigma, y):
    dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y))

def multi_tilted_loss(quantiles,y,f):
    '''
    Tilted loss function used
    for simultaneous generation of quantiles
    following (citation)
    '''
    #print y
    #print f
    # The term inside k.mean is a one line simplification of the first equation
    loss = 0



    # quantile loss
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k]-f[:,k])
        #loss += K.mean(K.maximum(q*e, (q-1)*e), axis=-1)
        loss += K.mean(q*e + K.clip(-e, K.epsilon(), np.inf), axis=-1)

    # crossing loss
    for i in range(len(quantiles)-1):
        q = quantiles[i]
        q1 = f[:,i]
        q2 = f[:,i+1]
        loss += K.mean(K.maximum(0.0, q1 - q2))

    return loss

class QuantileLossModel():
    '''
    Class storing both the first and second
    step models used for both experiementation
    and search analysis.
    Input:
        - Config dictionary
    Output:
        - 1st step model object
    '''

    def __init__(self, config):
        self._num_features = config['NUM_FEATURES']
        self._quantiles = config['QUANTILES']
        self._1_common_lstm_layer_num = config['1_COMMON_LSTM_NUM']
        self._2_common_lstm_layer_num = config['2_COMMON_LSTM_NUM']
        self._1_ind_lstm_layer_num = config['1_IND_LSTM_NUM']
        self._1_ind_dense_num = config['1_IND_DENSE_NUM']
        self._1_ind_dense_activation = config['1_IND_DENSE_ACTIVATION']
        self._1_out_dense_activation = config['1_OUT_DENSE_ACTIVATION']
        self._1_lr = config['1_LEARNING_RATE']
        self._pred_step = config['PRED_STEP']

    def create_linear_model(self):
        '''
        Creating the first step
        quantile loss model.
        '''
        tf.keras.backend.clear_session()

        tf.random.set_seed(1234)

        data_input = keras.Input(
            shape=(None, self._num_features), name="data_input"
        )

        out_layers = []
        for i in range(len(self._quantiles)):
            q = self._quantiles[i]
            out_layer = layers.Dense(1, activation = self._1_out_dense_activation, name="q{}".format(int(q*100)))(data_input)
            out_layers.append(out_layer)

        final_output = tf.concat(out_layers, 1)

        model = keras.Model(
            inputs=[data_input],
            outputs=[final_output]
        )

        model.compile(
        optimizer=keras.optimizers.Adam(self._1_lr),
        loss=lambda y,f: multi_tilted_loss(self._quantiles,y,f),
        metrics=['mae', 'mse']
        )

        return model

def train_1step_model(model_1, data, EPOCHS1):
    '''
    First model estimation
    '''
    history = model_1.fit(
        {"data_input": data['X_TRAIN']},
        {"tf.concat": data['Y_TRAINK']},
        validation_data=(data['X_TEST'], data['Y_TESTK']),
        epochs=EPOCHS1,
        batch_size=data['BATCH_SIZE'],
        verbose=2 #,
        )

    y_train_dist = model_1.predict(data['X_TRAIN'])
    y_test_dist = model_1.predict(data['X_TEST'])

    return y_train_dist, y_test_dist, history, model_1

def model_linear(param, data, EPOCHS1):
    '''
    First model creation
    '''
    model_creator_1 = QuantileLossModel(param)
    model_1 = model_creator_1.create_linear_model()
    return train_1step_model(model_1, data, EPOCHS1)