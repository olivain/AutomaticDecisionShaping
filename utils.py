
import cv2
import json
import os
import random


def serialize_image(image):
    serialized_image = image.tolist()
    return serialized_image

def write_candidate_to_json(candidate, filename):
    with open(filename, 'w') as json_file:
        json.dump(candidate, json_file)

def load_candidate_from_file(filename):
    with open(filename, 'r') as json_file:
        candidate = json.load(json_file)
        u = candidate["uuid"]
        file_path = f"candidates/{u}.png"
        image = cv2.imread(file_path)
    return image, candidate


def keep_three_pairs_random(folder_path):
    files = os.listdir(folder_path)
    file_names = []

    for file in files:
        print(file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension == '.png':
            file_names.append(file_name)
    random.shuffle(file_names)
    files_to_keep = 3

    for file_name in file_names:
        png_file = os.path.join(folder_path, file_name + ".png")
        json_file = os.path.join(folder_path, file_name + ".json")
        if files_to_keep > 0:
            if os.path.exists(png_file) and os.path.exists(json_file):
                files_to_keep -= 1
        else:
            print(f"[.] removing training data {file_name} ...")
            if os.path.exists(png_file):
                os.remove(png_file)
            if os.path.exists(json_file):
                os.remove(json_file)


def load_random_candidate(folder_path):
    files = os.listdir(folder_path)
    png_files = [file for file in files if file.lower().endswith('.png')]
    if not png_files:
        print("No PNG files found in the folder.")
        return None, None
    random_png_file = random.choice(png_files)
    file_name_without_extension = os.path.splitext(random_png_file)[0]
    file_path = os.path.join(folder_path, random_png_file)
    image = cv2.imread(file_path)
    filename = "candidates/" + file_name_without_extension + ".json"
    with open(filename, 'r') as json_file:
        json2 = json.load(json_file)

    return image, json2
