import tensorflow as tf
from efficientnet_lite import EfficientNetLiteB4
import os
import pickle

BATCH_SIZE = 32
TARGET_SIZE = (300, 300)


def preprocess_data(images, labels):
    images = (images - 127.00) / 128.00
    return images, labels


def augment_data(images, labels):
    return tf.image.random_flip_left_right(images), labels


def create_datasets(path):
    # Create tf.data.dataset objects
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=path,
        batch_size=BATCH_SIZE,
        image_size=TARGET_SIZE,
        label_mode="categorical",
        labels='inferred',
        seed=12341,
        validation_split=0.2,
        subset="training",
        color_mode="rgb"
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=path,
        batch_size=BATCH_SIZE,
        image_size=TARGET_SIZE,
        label_mode="categorical",
        labels='inferred',
        seed=12341,
        validation_split=0.2,
        subset="validation",
        color_mode="rgb"
    )

    return train_dataset, val_dataset


def build_model(num_classes=3):
    base_model = EfficientNetLiteB4(
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        include_top=False,
        pooling="avg",
        weights="imagenet"
    )

    base_model.trainable = False

    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])


def main_train(input_path, num_classes, model_output_path, epochs=5):
    train_dataset, val_dataset = create_datasets(input_path)
    # Add entropy to data and optimize it for training
    AUTOTUNE = tf.data.AUTOTUNE 
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=AUTOTUNE).map(
        augment_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_dataset = val_dataset.map(
        preprocess_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
    )

    model.save(model_output_path)
    

    labels = [i for i in os.listdir(input_path) if not os.path.isfile( os.path.join(input_path, i))]
    print(labels)
    pickle.dump(labels, open( os.path.join(model_output_path,'labels.pkl'), 'wb'))
    

#main_train("./yes", 3, "./die/my_new_model", 1)
