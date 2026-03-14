import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Dataset Path
data_path = "Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data"

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_path + "/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    data_path + "/val",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    data_path + "/test",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)


# Function to Train Models
def train_model(base_model, model_name):

    # Fine-tune last layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n==============================")
    print(f"Training {model_name}")
    print("==============================\n")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=8
    )
 
     # -------- Accuracy Plot --------

    plt.figure()

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(model_name + " Accuracy")
    plt.legend()

    plt.show()


# -------- Loss Plot --------

    plt.figure()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_name + " Loss")
    plt.legend()

    plt.show()
    
    loss, acc = model.evaluate(test_generator)

    print(f"\n{model_name} Test Accuracy:", acc)

    model.save(f"models/{model_name}.keras")

    print(f"{model_name} saved successfully\n")


# Train Models

train_model(
    VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "vgg16_model"
)

train_model(
    ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "resnet50_model"
)

train_model(
    MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "mobilenet_model"
)

train_model(
    InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "inception_model"
)

train_model(
    EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "efficientnet_model"
)