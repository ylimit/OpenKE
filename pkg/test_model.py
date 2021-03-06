from keras.layers import Input, Dense
from keras.models import Model
from keras import backend
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np


class TestModel(object):
    def __init__(self, dims):
        dim_in = dims[0]
        dim_out = dims[-1]
        dims_encoder = dims[1:]
        dims_decoding = dims[:-1]
        dims_decoding.reverse()

        encoder_input = Input(shape=(dim_in,), name='EncoderIn')
        decoder_input = Input(shape=(dim_out,), name='DecoderIn')

        encoded = encoder_input

        # Construct encoder layers
        for i, dim in enumerate(dims_encoder):
            name = 'Encoder{0}'.format(i)
            encoded = Dense(dim, activation='relu', name=name)(encoded)

        # Construct decoder layers
        # The decoded is connected to the encoders, whereas the decoder is not
        decoded = encoded
        decoder = decoder_input

        name = 'decoder'
        layer = Dense(1, activation='sigmoid', name=name)

        decoded = layer(decoded)
        decoder = layer(decoder)

        self.autoencoder = Model(inputs=encoder_input, outputs=decoded)
        self.encoder = Model(inputs=encoder_input, outputs=encoded)
        self.decoder = Model(inputs=decoder_input, outputs=decoder)

        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, x, y, epochs, batch_size, log_dir='/tmp/autoencoder', stop_early=True):
        callbacks = []
        if backend._BACKEND == 'tensorflow':
            callbacks.append(TensorBoard(log_dir=log_dir))

        if stop_early:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto'))

        n_samples = len(x)
        n_test = int(n_samples * 0.1)
        print("Total samples: %d, test: %d" % (n_samples, n_test))
        x_test, x_train = x[:n_test], x[n_test:]
        y_test, y_train = y[:n_test], y[n_test:]

        self.autoencoder.fit(x_train, y_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_split=0.1,
                             callbacks=callbacks)

        from sklearn.metrics import precision_recall_fscore_support
        y_predict = self.autoencoder.predict(x_test)
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict <= 0.5] = 0
        print(precision_recall_fscore_support(y_test, y_predict))

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def summary(self):
        self.autoencoder.summary()
