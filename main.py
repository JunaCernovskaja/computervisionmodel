import os
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
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

def train_model(model, train_generator, validation_generator, epochs, checkpoint_dir, history_path):
    checkpoint_filepath = os.path.join(checkpoint_dir, 'main_training-{epoch:02d}-{val_loss:.2f}.keras')
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=1)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )


    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {history_path}")

    return history

def main():
    try:
        print("Starting the main dataset training...")


        base_dir = 'C:/Users/Vartotojas/Desktop/project'
        train_dir = os.path.join(base_dir, 'dataset/train')


        labels_file = os.path.join(train_dir, 'labels.csv')
        try:
            train_df = pd.read_csv(labels_file)
            train_df = shuffle(train_df)
            print(f"Loaded and shuffled training data from {labels_file}")
        except FileNotFoundError:
            print(f"Error: File {labels_file} not found.")
            return
        except Exception as e:
            print(f"Error occurred while loading data: {e}")
            return


        img_width, img_height = 224, 224
        batch_size = 32
        main_training_epochs = 50


        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )


        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory=train_dir,
            x_col='filename',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )


        validation_generator = train_datagen.flow_from_dataframe(
            train_df,
            directory=train_dir,
            x_col='filename',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )


        checkpoint_dir = os.path.join(base_dir, 'checkpoints')
        create_directory(checkpoint_dir)


        history_path = os.path.join(base_dir, 'training_history.json')


        fine_tuned_model_path = os.path.join(base_dir, 'fine_tuned_war_photography_classifier_efficientnet.keras')
        model = load_model(fine_tuned_model_path)
        print(f"Loaded fine-tuned model from {fine_tuned_model_path}")


        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        print("Training with the main dataset...")
        train_model(model, train_generator, validation_generator, main_training_epochs, checkpoint_dir, history_path)


        final_model_path = os.path.join(base_dir, 'final_war_photography_classifier_efficientnet.keras')
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        print("Main dataset training completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()