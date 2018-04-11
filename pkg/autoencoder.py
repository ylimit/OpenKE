from keras.layers import Input, Dense
from keras.models import Model
from keras import backend
from keras.callbacks import EarlyStopping, TensorBoard


class DeepAutoencoder(object):
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
        for i, dim in enumerate(dims_decoding):
            name = 'Decoder{0}'.format(i)

            activation = 'relu'
            if i == len(dims_decoding) - 1:
                activation = 'sigmoid'

            layer = Dense(dim, activation=activation, name=name)

            decoded = layer(decoded)
            decoder = layer(decoder)

        self.autoencoder = Model(inputs=encoder_input, outputs=decoded)
        self.encoder = Model(inputs=encoder_input, outputs=encoded)
        self.decoder = Model(inputs=decoder_input, outputs=decoder)

        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, x_train, epochs, batch_size, log_dir='/tmp/autoencoder', stop_early=True):
        callbacks = []
        if backend._BACKEND == 'tensorflow':
            callbacks.append(TensorBoard(log_dir=log_dir))

        if stop_early:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto'))

        self.autoencoder.fit(x_train, x_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_split=0.1,
                             callbacks=callbacks)

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def summary(self):
        self.autoencoder.summary()
