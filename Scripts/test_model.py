import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def test_model():
    base_dir = 'C:/Users/Vartotojas/Desktop/project'
    test_dir = os.path.join(base_dir, 'dataset/test')
    labels_file = os.path.join(test_dir, 'labels.csv')

    try:

        test_df = pd.read_csv(labels_file)
        print(f"Loaded test data from {labels_file}")


        img_width, img_height = 224, 224
        batch_size = 32


        test_datagen = ImageDataGenerator(rescale=1.0/255)


        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            directory=test_dir,
            x_col='filename',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )


        model_path = os.path.join(base_dir, 'final_war_photography_classifier_efficientnet.keras')
        model = load_model(model_path)
        print(f"Loaded final trained model from {model_path}")


        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_model()