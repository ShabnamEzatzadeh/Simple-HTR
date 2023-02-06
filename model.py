from tensorflow import keras
from tensorflow.keras import layers
from ctcLayer import CTCLayer


def build_model(img_width, img_height, max_length, characters):
    # Inputs to the model
    input_img = layers.Input(shape=(img_width, img_height, 1),
                             name='input_data',
                             dtype='float32')
    labels = layers.Input(name='input_label', shape=[max_length], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    conv1 = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='SAME', name='Conv1')(input_img)
    max1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)
    conv2 = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='SAME', name='Conv2')(max1)
    max2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(conv2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME', name='Conv3')(max2)
    max3 = keras.layers.MaxPooling2D(pool_size=(1, 2), name="pool3")(conv3)

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(max3)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)

    # Predictions
    x = layers.Dense(len(characters) + 1, activation='softmax', name='dense2')(x)

    # Calculate CTC
    output = CTCLayer(name='ctc_loss')(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(inputs=[input_img,
                                       labels,
                                       input_length,
                                       label_length],
                               outputs=output,
                               name='ocr_model_v1')

    # Optimizer
    sgd = keras.optimizers.SGD(learning_rate=0.002,
                               decay=1e-6,
                               momentum=0.9,
                               nesterov=True,
                               clipnorm=5)

    # Compile the model and return
    model.compile(optimizer=sgd)
    return model
