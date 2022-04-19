import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from keras.layers import merge

class WeightsUpdater(tf.keras.callbacks.Callback):
    def __init__(self, alpha, beta, epochs):
        self.alpha = alpha
        self.beta = beta
        self.total_epochs = epochs
        self.coef = self.alpha/self.total_epochs
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch > (int(self.total_epochs*0.9)):
            self.alpha = 0.05
            self.beta = 0.95
            
        print (f"\nLoss_weights used:", K.get_value(self.alpha), K.get_value(self.beta))

def ls_quantile_cdf(mu, sigma, skewness, tailweight, quantile_values, quantiles):
    loss = 0
    dist = tfp.distributions.SinhArcsinh(loc=mu, scale=sigma, skewness=skewness, tailweight=tailweight)
    q_val = tf.squeeze(quantile_values)
    q_est = dist.quantile(quantiles)
    loss = K.mean(tf.math.squared_difference(q_val, q_est), axis=-1)
    return loss

def multi_tilted_loss_v2(quantiles,y,f):
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
        e = (y-f[:,k])
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

    def create_deepgar_skew_v2(self):
        '''
        Creating the first step
        quantile loss model.
        '''
        tf.keras.backend.clear_session()

        tf.random.set_seed(1234)

        data_input = keras.Input(
            shape=(None, self._num_features), name="data_input"
        )

        lstm_origin = layers.LSTM(self._1_common_lstm_layer_num, return_sequences=True)(data_input)
        #lstm_lat_factors = layers.LSTM(self._2_common_lstm_layer_num, return_sequences=True)(lstm_origin)

        out_layers = []
        for i in range(len(self._quantiles)):
            q = self._quantiles[i]
            lstm_layer1 = layers.Dense(self._1_ind_lstm_layer_num, activation = "tanh")(lstm_origin)
            dense_layer = layers.Dense(self._1_ind_dense_num, activation=self._1_ind_dense_activation)(lstm_layer1)
            out_layer = layers.Dense(1, activation = self._1_out_dense_activation, name="q{}".format(int(q*100)))(dense_layer)
            out_layers.append(out_layer)

        quantiles_layer = tf.concat(out_layers, 1)

        lstm_layer2 = layers.LSTM(self._1_ind_lstm_layer_num)(lstm_origin)

        mu = layers.Dense(1, activation="linear", name='mu')(lstm_layer2)
        sigma = layers.Dense(1, activation=lambda x: tf.nn.elu(x)+1, name='sigma')(lstm_layer2)
        skewness = layers.Dense(1, activation="linear", name='skewness')(lstm_layer2)
        tailweight = layers.Dense(1, activation=lambda x: tf.nn.elu(x)+1, name='tailweight')(lstm_layer2)

        y_real = keras.Input(shape=(1, ))
        lossF = ls_quantile_cdf(mu,sigma,skewness,tailweight,quantiles_layer,self._quantiles)
        lossQ= multi_tilted_loss_v2(self._quantiles,y_real,quantiles_layer)

        model = keras.Model(
            inputs=[data_input, y_real],
            outputs=[mu, sigma, skewness, tailweight]
        )

        final_loss = [lossQ, lossF]

        model.add_loss(final_loss)

        alpha = K.variable(0.95)
        beta = K.variable(0.05)

        model.compile(
        optimizer=keras.optimizers.Adam(self._1_lr),
        loss_weights=[alpha, beta]
        )

        return model, alpha, beta

def train_tfp_model_norm2(model_1, data, EPOCHS1, CALLBACKS):
    '''
    First model estimation
    '''
    history = model_1.fit(
        [data['X_TRAIN'], data['Y_TRAIN']],
        validation_data=[[data['X_TEST'], data['Y_TEST']]],
        epochs=EPOCHS1,
        batch_size=data['BATCH_SIZE'],
        verbose=1,
        callbacks=[CALLBACKS]
        )

    y_train_dist = model_1.predict([data['X_TRAIN'],data['Y_TRAIN']])
    y_test_dist = model_1.predict([data['X_TEST'],data['Y_TEST']])

    return y_train_dist, y_test_dist, history, model_1

def model_deepgar_skew_v2(param, data, EPOCHS1, patience):
    '''
    LSTM version 1
    '''
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model_creator_1 = QuantileLossModel(param)
    model_1, alpha, beta = model_creator_1.create_deepgar_skew_v2()
    return train_tfp_model_norm2(model_1, data, EPOCHS1, [WeightsUpdater(alpha, beta, EPOCHS1), early_stopper])
