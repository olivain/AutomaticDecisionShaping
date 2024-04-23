import numpy as np
import cv2
import random
import os



def stack_images_on_top(existing_file, new_image):
    existing_image = cv2.imread(existing_file, cv2.IMREAD_GRAYSCALE)
    if existing_image is None:
        existing_image = np.ones_like(new_image)

    if len(existing_image.shape) > 2:
        existing_image = cv2.cvtColor(existing_image, cv2.COLOR_BGR2GRAY)
    if len(new_image.shape) > 2:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    stacked_image = np.vstack((new_image, existing_image))
    cv2.imwrite(existing_file, stacked_image)



def display_spaced_stacked_copies(image_file, num_copies, orientation='horizontal', target_width=800, column_spacing_cm=2):
    tmp = cv2.imread(image_file)
    _tmph, _tmpw = tmp.shape[:2]
    blank_space = (255, 255, 255) 
    blank_space_image = np.full((_tmph, int(column_spacing_cm * _tmpw / 2.54), 3), blank_space, dtype=tmp.dtype)
    
    stacked_images = []
    for i in range(num_copies):
        imfile = f"columns/{i}.png"
        image = cv2.imread(imfile)
    
        if image is None:
            print("Error: Unable to load the image.")
            return
        
        stacked_images.append(image)
        if i < num_copies - 1:
            stacked_images.append(blank_space_image)
    
    if orientation == 'horizontal':
        stacked_image = np.hstack(stacked_images)
    elif orientation == 'vertical':
        stacked_image = np.vstack(stacked_images)
    
    aspect_ratio = stacked_image.shape[1] / stacked_image.shape[0]
    target_height = int(target_width / aspect_ratio)
    stacked_image_resized = cv2.resize(stacked_image, (target_width, target_height))

    return stacked_image_resized
