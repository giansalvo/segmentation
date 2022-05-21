import tensorflow as tf
import tensorflow_datasets as tfds

from IPython.display import clear_output
import matplotlib.pyplot as plt
from time import gmtime, strftime

from unet2 import create_model_UNet2


# network model parameters
OUTPUT_CLASSES = 3
INPUT_SIZE = 128

# training parameters
EPOCHS = 20
BATCH_SIZE = 64
VAL_SUBSPLITS = 5
BUFFER_SIZE = 1000
DEBUG = False

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (INPUT_SIZE, INPUT_SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (INPUT_SIZE, INPUT_SIZE))
  input_image, input_mask = normalize(input_image, input_mask)
  return input_image, input_mask


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
  
  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels


def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])
            

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if DEBUG:
      clear_output(wait=True)
      print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
      show_predictions()


##########################
### MAIN STARTS HERE
##########################
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
print(dataset)

TRAIN_LENGTH = info.splits['train'].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

if DEBUG:
  print("Dataset ready to train, let's show some examples to check that everything is ok...")
  for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

print("create network model...")
model = create_model_UNet2(output_channels=OUTPUT_CLASSES, input_size=INPUT_SIZE)
print("compile network model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

callbacks = [
          DisplayCallback(),
          tf.keras.callbacks.ModelCheckpoint("unet2.h5", 
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True)
]

print("start training network...")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=callbacks)
print("training network end.")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

print("plotting training history...")
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
# plt.ylim([0, 1])
plt.legend()
plt.show()

print("show some predictions...")
show_predictions(test_batches, 3)

print("program terminated correctly.")
