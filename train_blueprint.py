# train_blueprint.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_sketch_classifier(input_shape=(64, 64, 1), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    rgb_inputs = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(64, 64, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = base_model(rgb_inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)