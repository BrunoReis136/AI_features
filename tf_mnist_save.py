import tensorflow as tf

def model_mnist():
  (x_train, y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

  x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
  x_test = x_test[..., tf.newaxis].astype("float32") / 255.0

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
      tf.keras.layers.MaxPooling2D((2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
  ])

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  model.compile(optimizer="adam",
                loss=loss_fn,
                metrics=["accuracy"])

  prediction_single_sample = model(x_train[:1])

  probabilities_single_sample = tf.nn.softmax(prediction_single_sample).numpy()
  print(f'Previsões previstas para a primeira amostra: {probabilities_single_sample}')

  loss_single_sample = loss_fn(y_train[:1],prediction_single_sample).numpy()
  print(f'Perda para a primeira amostra: {loss_single_sample}')

  class myCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.final_acc = None

    def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}
      acc = logs.get('accuracy')

      if acc is not None and acc > 0.99 :
          print('\nValores de Precisão ou Perda atingidos.')
          self.final_acc = acc
          self.model.stop_training = True

  callback = myCallback()
  
  print('Iniciando treinamento do modelo...')
  model.fit(x_train, y_train, epochs=8, callbacks=[callback])

  print('Avaliando modelo...')
  model.evaluate(x_test, y_test, verbose=2)

  salva_sn = input('Deseja salvar o modelo treinado?(s/n)').lower().strip()

  if salva_sn == 's':
    modelo_savepath = f'model_cnn({round(callback.final_acc,3)*100}).keras'

    try:
      model.save(modelo_savepath)
      print('Modelo salvo com csucesso.')
    except Exception as e:
      print(f'Erro ao salvar modelo: {e}')
  else:
    print('Modelo não salvo.')

model_mnist()
