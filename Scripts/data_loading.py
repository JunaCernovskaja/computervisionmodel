import os
import pandas as pd

def create_labels_csv(base_dir, output_csv='labels.csv'):

    train_dir = os.path.join(base_dir, 'train')


    data = []


    for category in os.listdir(train_dir):
        category_dir = os.path.join(train_dir, category)
        if os.path.isdir(category_dir):

            for file_name in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file_name)
                if os.path.isfile(file_path):

                    data.append({'filename': os.path.relpath(file_path, train_dir), 'label': category})


    df = pd.DataFrame(data)


    csv_path = os.path.join(train_dir, output_csv)
    df.to_csv(csv_path, index=False)
    print(f"Created labels.csv file at {csv_path}")

if __name__ == "__main__":

    base_dir = 'C:/Users/Vartotojas/Desktop/project/dataset'
    create_labels_csv(base_dir)