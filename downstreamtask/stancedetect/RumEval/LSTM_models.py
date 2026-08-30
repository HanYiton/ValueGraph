import numpy as np
from keras.layers import Input, LSTM, Dense, Dropout, Masking, Concatenate
from keras.models import Model
from keras import optimizers, regularizers

def LSTM_model_stance(x_train, y_train, x_test, user_features_train, user_features_test, params, eval=False):
    """
    LSTM branch-level + User Features
    x_train: (num_branches, max_branch_len, embedding_dim)
    user_features_train: (num_branches, user_feature_dim)
    """

    num_lstm_units = int(params['num_lstm_units'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']

    # branch input
    branch_input = Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = Masking(mask_value=0.)(branch_input)
    x = LSTM(num_lstm_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(num_dense_units, activation='relu')(x)
    x = Dropout(0.5)(x)

    # user feature input
    user_input = Input(shape=(user_features_train.shape[1],))
    concat = Concatenate()([x, user_input])

    out = Dense(4, activation='softmax',
                activity_regularizer=regularizers.l2(l2reg))(concat)

    model = Model(inputs=[branch_input, user_input], outputs=out)

    adam = optimizers.Adam(learning_rate=learn_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([x_train, user_features_train], y_train,
              batch_size=mb_size,
              epochs=num_epochs,
              shuffle=True,
              verbose=1)

    pred_probabilities = model.predict([x_test, user_features_test], batch_size=mb_size, verbose=1)
    confidence = np.max(pred_probabilities, axis=1)
    y_pred = np.argmax(pred_probabilities, axis=1)

    return y_pred, confidence
