import tensorflow as tf
from callback import myCallback


def train(model, train_data_generator):
    # Add early stopping
    es = myCallback()
    # Create a callback that saves the model's weights every 50 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="../checkpoints/cp-20-new.ckpt",
        verbose=1,
        save_weights_only=True,
        save_freq=100)

    # Train the model
    history = model.fit(train_data_generator, epochs=500, callbacks=[es, cp_callback])
    model.save("../models/train-keras-ours-new.h5", overwrite=True)
