import numpy as np
import tensorflow as tf
import cv2
import random
from scipy.spatial import Delaunay
import uuid
import json
import utils
import candidaters

# Fonction pour extraire les caractéristiques des images
def extract_features(image, json):
    num_sides = json["nb_sides"] 
    type_nb = json["type_nb"] 
    black_percentage = json["black_percentage"] 
    nb_elected = json["nb_elected"] 
    im_mean = np.mean(image) 
     
    box = json["box"] #[x,y,w,h]
    box_features = [
        box[0],  # x-coordinate
        box[1],  # y-coordinate
        box[2],  # width
        box[3],  # height
    ]
    
    # Concatenate all features into a single list
    return [num_sides, type_nb, black_percentage, nb_elected, im_mean] + box_features

def train(nb_models, type_determinant=1, nb_epoch=10, nb_data=1000 ):
    X_train = []
    y_train = []
    models_y = []

    trained_model = []

    mean_features = 0
    # print("TRAINING TO INFLUENCE CHOICE ON TYPE ",type_determinant)

    for _ in range(nb_data):  
        
        json2, image2 = candidaters.generate_new_candidate()
        image1, json1 = utils.load_random_candidate("candidates")
      
        features1 = extract_features(image1,json1)
        # print(json1)
        features2 = extract_features(image2,json2)
        # print(json2)

        X_train.append(features1)
        X_train.append(features2)

        for nb in range(nb_models):
            if nb == 0:
                print(" -| tendance 0")
                # tendance a voter pour la forme ayant le plus de cotes: (nb_sides)
                label1 = 1 if features1[0] > features2[0] else 0
                label2 = 1 if features2[0] > features1[0] else 0
            elif nb == 1:
                print(" -| tendance 1")
                # tendance a voter pour le moins elu historiquement (nb_elected)
                label1 = 1 if features1[3] < features2[3] else 0
                label2 = 1 if features2[3] < features1[3] else 0
            elif nb == 2:
                print(" -| tendance 2")
                # #tendance a voter pour les plus grosses formes: (black_percentage)
                label1 = 1 if features1[2] > features2[2] else 0
                label2 = 1 if features2[2]  > features1[2] else 0
            elif nb == 3:
                print(" -| tendance 3")
                # #tendance aleatoire ??
                label1 = 1 if  random.choice([True, False]) else 0
                label2 = 1 if random.choice([True, False]) else 0
            elif nb == 4:
                print(" -| tendance 4")
                # tendance a voter pour un certain type de forme: (type_nb)
                label1 = 1 if features1[1] == type_determinant else 0
                label2 = 1 if features2[1] == type_determinant else 0
                print("type_determinant: ",type_determinant)
            elif nb == 5:
                print(" -| tendance 5")
                #tendance a voter pour les formes plus differentes de la moyenne.
                diviser = 3
                if mean_features==0:
                    mean_features = features1[4]
                    diviser = 2
                distance1 = abs(mean_features - features1[4])
                distance2 = abs(mean_features - features2[4])
                label1 = 1 if distance1 > distance2 else 0
                label2 = 1 if distance2  > distance1 else 0
                mean_features = (mean_features + features2[4])/diviser

            elif nb == 6:
                print(" -| tendance 6")
                #tendance a voter pour un certain type de forme: (type_nb)
                new_deter = type_determinant
                if type_determinant > 1:
                    new_deter = type_determinant - 1
                else:
                    new_deter = type_determinant+1
                label1 = 1 if features1[1] == new_deter else 0
                label2 = 1 if features2[1] == new_deter else 0
            elif nb == 7:
                # golden ratio
                print(" -| tendance 7")
                #tendance a voter pour les formes les plus proches de la proportion d'or: (box)
                r1 = candidaters.get_distance_golden_ratio(json1["box"][0],json1["box"][1],json1["box"][2],json1["box"][3])
                r2 =  candidaters.get_distance_golden_ratio(json2["box"][0],json2["box"][1],json2["box"][2],json2["box"][3])
                label1 = 1 if r1 < r2 else 0
                label2 = 1 if r2 < r1 else 0
            elif nb == 8:
                print(" -| tendance 8")
                # #tendance aleatoire ??
                label1 = 1 if  random.choice([True, False]) else 0
                label2 = 1 if random.choice([True, False]) else 0
            
            if len(models_y) <= nb_models:
                models_y.append([])

            models_y[nb].extend([label1, label2])

    X_train = np.array(X_train)

    for i in range(nb_models):
        y_train = np.array(models_y[i])

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=nb_epoch, batch_size=32)
        trained_model.append(model)
    return trained_model

def agent_decision(model, image1, json1, image2, json2):
    features1 = extract_features(image1, json1)
    features2 = extract_features(image2, json2)
    decision1 = model.predict(np.array([features1]))[0][0]
    decision2 = model.predict(np.array([features2]))[0][0]
    return decision1, decision2
