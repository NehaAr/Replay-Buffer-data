import tensorflow as tf
from tensorflow.keras import optimizers, initializers, layers,models

def create_model(num_class,input_dim=68, hidden_layers=[128, 64], learning_rate=0.001):
  model=models.Sequential()
  model.add(layers.Input(shape=(input_dim,)))
  for neurons in hidden_layers:
    model.add(layers.Dense(neurons,activation='relu',kernel_initializer=initializers.GlorotUniform()))
  model.add(layers.Dense(num_class,activation='softmax'))
  optimizer=optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
  return model
