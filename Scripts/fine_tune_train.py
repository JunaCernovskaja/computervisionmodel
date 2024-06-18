import os
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def load_labels(base_dir, filename='labels.csv'):
    labels_path = os.path.join(base_dir, filename)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"The file {labels_path} does not exist.")
    return pd.read_csv(labels_path)

def fine_tune_model(model, train_generator, validation_generator, epochs, checkpoint_dir):
    checkpoint_filepath = os.path.join(checkpoint_dir, 'fine_tune-{epoch:02d}-{val_loss:.2f}.weights.h5')
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=1)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    return history

def main():
    try:
        print("Starting the fine-tuning process...")


        base_dir = 'C:/Users/Vartotojas/Desktop/project'
        handpicked_dir = os.path.join(base_dir, 'handpicked_dataset/train')


        handpicked_labels_file = os.path.join(handpicked_dir, 'labels.csv')
        try:
            handpicked_train_df = pd.read_csv(handpicked_labels_file)
            handpicked_train_df = shuffle(handpicked_train_df)
            print(f"Loaded and shuffled handpicked training data from {handpicked_labels_file}")
        except FileNotFoundError:
            print(f"Error: File {handpicked_labels_file} not found.")
            return
        except Exception as e:
            print(f"Error occurred while loading handpicked data: {e}")
            return


        img_width, img_height = 224, 224
        batch_size = 32
        fine_tune_epochs = 20


        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )


        handpicked_train_generator = train_datagen.flow_from_dataframe(
            handpicked_train_df,
            directory=handpicked_dir,
            x_col='filename',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_dataframe(
            handpicked_train_df,
            directory=handpicked_dir,
            x_col='filename',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )


        checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        create_directory(checkpoint_dir)


        fine_tuned_model_path = os.path.join(base_dir, 'fine_tuned_war_photography_classifier_efficientnet.keras')
        model = load_model(fine_tuned_model_path)
        print(f"Loaded fine-tuned model from {fine_tuned_model_path}")


        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        print("Fine-tuning with the handpicked dataset again...")
        fine_tune_model(model, handpicked_train_generator, validation_generator, fine_tune_epochs, checkpoint_dir)


        fine_tuned_model_path = os.path.join(base_dir, 'fine_tuned_war_photography_classifier_efficientnet.keras')
        model.save(fine_tuned_model_path)
        print(f"Fine-tuned model saved to {fine_tuned_model_path}")

        print("Fine-tuning completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
