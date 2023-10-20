from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from keras.initializers import RandomUniform
from keras.utils import plot_model
import tensorflow as tf

def get_model(words_input_dim, words_output_dim, words_weights, case_input_dim, case_output_dim, case_weights,
              char_2_idx_len, label_2_idx_len):
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=words_input_dim, output_dim=words_output_dim, weights=[words_weights],
                      trainable=False)(words_input)
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=case_input_dim, input_dim=case_output_dim, weights=[case_weights],
                       trainable=False)(casing_input)
    character_input = Input(shape=(None, 52,), name='char_input')
    embed_char_out = TimeDistributed(
        Embedding(char_2_idx_len, 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name='char_embedding')(
        character_input)
    dropout = Dropout(0.5)(embed_char_out)
    conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(
        dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)
    output = concatenate([words, casing, char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(label_2_idx_len, activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()

    return model
