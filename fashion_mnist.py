import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images[..., tf.newaxis].astype("float32") / 255.0
test_images = test_images[..., tf.newaxis].astype("float32") / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28,28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs['accuracy'] > 0.9:
      print('\n Accuracy enought, so canceling the training...')
      self.model.stop_training = True


model.fit(train_images, train_labels, epochs=10, callbacks=[myCallback()])

model.evaluate(test_images, test_labels)
