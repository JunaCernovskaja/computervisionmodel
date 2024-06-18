import os
import json
import matplotlib.pyplot as plt

def plot_training_history(history_path):
    if not os.path.isfile(history_path):
        print(f"Error: Training history file not found at {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)


    plt.figure(figsize=(12, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    base_dir = 'C:/Users/Vartotojas/Desktop/project'
    history_path = os.path.join(base_dir, 'training_history.json')

    plot_training_history(history_path)

if __name__ == "__main__":
    main()