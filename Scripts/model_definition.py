from tensorflow.keras import layers, models
from efficientnet.tfkeras import EfficientNetB0

def build_model(input_shape=(224, 224, 3), num_classes=18):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model