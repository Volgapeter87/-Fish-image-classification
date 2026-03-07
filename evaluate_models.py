import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# Dataset path
data_path = "Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data"

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    data_path + "/test",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())

models_path = "models"

for model_file in os.listdir(models_path):

    print("\n==============================")
    print("Evaluating:", model_file)
    print("==============================")

    model = tf.keras.models.load_model(os.path.join(models_path, model_file))

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)

    print("\nClassification Report\n")
    print(classification_report(test_generator.classes, y_pred, target_names=class_labels))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(test_generator.classes, y_pred))