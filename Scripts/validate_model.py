import os
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def validate_model():
    base_dir = 'C:/Users/Vartotojas/Desktop/project'
    validation_dir = os.path.join(base_dir, 'dataset/train')
    labels_file = os.path.join(validation_dir, 'labels.csv')

    try:

        validation_df = pd.read_csv(labels_file)
        validation_df = shuffle(validation_df)
        print(f"Loaded and shuffled validation data from {labels_file}")


        img_width, img_height = 224, 224
        batch_size = 32


        validation_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)


        validation_generator = validation_datagen.flow_from_dataframe(
            validation_df,
            directory=validation_dir,
            x_col='filename',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )


        model_path = os.path.join(base_dir, 'final_war_photography_classifier_efficientnet.keras')
        model = load_model(model_path)
        print(f"Loaded final trained model from {model_path}")


        validation_loss, validation_accuracy = model.evaluate(validation_generator)
        print(f"Validation Loss: {validation_loss}")
        print(f"Validation Accuracy: {validation_accuracy}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    validate_model()