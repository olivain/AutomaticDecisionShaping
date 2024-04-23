import numpy as np
import cv2
import random
from scipy.spatial import Delaunay
import utils
import uuid

def generate_uuid():
    return str(uuid.uuid4().hex)

def generate_connected_shape_image2(width=400, height=400, min_points=5, max_points=20):
    points = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(random.randint(min_points, max_points))]
    tri = Delaunay(points)
    return cv2.fillPoly(np.ones((height, width), dtype=np.uint8) * 255, [np.array([points[i] for i in simplex], dtype=np.int32) for simplex in tri.simplices], 0)

def generate_connected_shape_image(width=400, height=400, min_points=3, max_points=15):
    points = [(random.randint(10, width - 10), random.randint(10, height - 10)) for _ in range(random.randint(min_points, max_points))]
    tri = Delaunay(points)
    image = np.ones((height, width), dtype=np.uint8) * 255
    image = cv2.polylines(image, [np.array([points[i] for i in simplex], dtype=np.int32) for simplex in tri.simplices], True, 0, thickness=3)
    return cv2.fillPoly(image, [np.array([points[i] for i in simplex], dtype=np.int32) for simplex in tri.simplices], 0)

def generate_connected_shape_image_xp(width=400, height=400, min_points=3, max_points=15):
    mid = (width / 2, height / 2)
    first_point = (random.randint(10, width - 10), random.randint(10, height - 10))
    second_point = (first_point[0] - 190 if first_point[0] > mid[0] else first_point[0] + 190,
                    first_point[1] - 190 if first_point[1] > mid[1] else first_point[1] + 190)
    third_point = (mid[0] - random.randint(100, mid[0] - 10) if second_point[0] > mid[0] else mid[0] + random.randint(170, 190),
                   mid[1] - random.randint(100, mid[1] - 10) if second_point[1] > mid[1] else mid[1] + random.randint(170, 190))
    
    points = [first_point, second_point, third_point]
    points += [(random.randint(10, width - 10), random.randint(10, height - 10)) for _ in range(random.randint(min_points - 2, max_points - 2))]
    
    tri = Delaunay(points)
    image = np.ones((height, width), dtype=np.uint8) * 255
    image = cv2.polylines(image, [np.array([points[i] for i in simplex], dtype=np.int32) for simplex in tri.simplices], True, 0, thickness=3)
    return cv2.fillPoly(image, [np.array([points[i] for i in simplex], dtype=np.int32) for simplex in tri.simplices], 0)


def generate_random_shape_image_one(width=400, height=400, min_shape_size=150, max_shape_size=190):
    s = random.randint(3, 20)
    return cv2.fillPoly(np.ones((height, width), dtype=np.uint8) * 255, [np.array([(int(width / 2 + random.randint(min_shape_size, max_shape_size) * np.cos(i * 2 * np.pi / s)), int(height / 2 + random.randint(min_shape_size, max_shape_size) * np.sin(i * 2 * np.pi / s))) for i in range(s)])], 0) if s else np.ones((height, width), dtype=np.uint8) * 255

def count_perimeter_lines(image):
    def get_angle(p1, p2, p3):
        v1 = np.subtract(p2, p1)
        v2 = np.subtract(p2, p3)
        cos = np.inner(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        return np.rad2deg(rad)

    def get_angles(p, d):
        n = len(p)
        return [(p[i], get_angle(p[(i-d) % n], p[i], p[(i+d) % n])) for i in range(n)]

    def remove_straight(p):
        angles = get_angles(p, 2)
        return [p for (p, a) in angles if a < 170]

    def max_corner(p):
        angles = get_angles(p, 1)
        j = 0
        while j < len(angles):
            k = (j + 1) % len(angles)
            (pj, aj) = angles[j]
            (pk, ak) = angles[k]
            if np.linalg.norm(np.subtract(pj, pk)) <= 4:
                if aj > ak:
                    angles.pop(j)
                else:
                    angles.pop(k)
            else:
                j += 1
        return [p for (p, a) in angles]

    perimeter_lines = 1
    ret, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0]) 
    
    for c in contours:
        pts = [v[0] for v in c]
        pts = remove_straight(pts)
        pts = max_corner(pts)
        angles = get_angles(pts, 1)
        prev_point = None
        for (p, a) in angles:
            if prev_point is not None and np.linalg.norm(np.subtract(p, prev_point)) >= 10:
                perimeter_lines += 1
            prev_point = p
    return perimeter_lines, x, y, w, h

def calculate_black_white_percentage(image):
    total_pixels = image.size
    black_pixels = np.count_nonzero(image == 0)
    white_pixels = np.count_nonzero(image == 255)
    black_percentage = (black_pixels / total_pixels) * 100
    white_percentage = (white_pixels / total_pixels) * 100
    return black_percentage, white_percentage


def get_distance_golden_ratio(x,y,w,h):
    x1, y1, x2, y2 = x, y, x+w, y+h 
    width_shape = x2 - x1
    height_shape = y2 - y1
    part1 = max(width_shape, height_shape)
    part2 = min(width_shape, height_shape)
    ratio = part1 / part2
    golden_ratio = (1 + np.sqrt(5)) / 2
    distance = abs(ratio - golden_ratio)
    return distance


def generate_new_candidate(training=False):
    random_number = random.randint(1, 3)

    if random_number == 1:
        image = generate_connected_shape_image_xp()
    elif random_number == 2:
        image = generate_connected_shape_image()
    else:
        image = generate_random_shape_image_one()

    #generate metadatas
    nb_sides, x, y, w, h = count_perimeter_lines(image)
    #type of shape 
    type_nb = random_number
    #black area
    black_percentage, _ = calculate_black_white_percentage(image)
    #nb elections success
    nb_elected = 0
    if training == True:
        nb_elected = random.randint(0, 100)
    
    random_boolean = random.choice([True, False])
    if random_boolean:
        nb_elected = random.randint(0, 50)
    uuid = generate_uuid()

    file_path = "candidates/"+uuid+".png"
    cv2.imwrite(file_path, image)

    candidate = {
    "img": file_path,
    "nb_sides": nb_sides,
    "type_nb": type_nb,
    "black_percentage": black_percentage,
    "nb_elected":nb_elected,
    "uuid":uuid,
    "box": [x, y, w, h]
    }
    
    json_file_name = "candidates/"+uuid + ".json"
    utils.write_candidate_to_json(candidate,json_file_name)
    im, candidate_from_file = utils.load_candidate_from_file(json_file_name)

    return candidate_from_file, im

def create_text_image_underneath(base_image, text, y_position):
    stacked_image = base_image.copy()

    if len(base_image.shape) > 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (stacked_image.shape[1] - text_size[0]) // 2
    text_y = y_position + 30
    text_height = 40
    result_height = stacked_image.shape[0] + text_height

    if len(stacked_image.shape) == 2:
        extended_image = np.ones((result_height, stacked_image.shape[1]), dtype=np.uint8) * 255
    else: 
        extended_image = np.ones((result_height, stacked_image.shape[1], stacked_image.shape[2]), dtype=np.uint8) * 255

    extended_image[:stacked_image.shape[0], :stacked_image.shape[1]] = stacked_image

    cv2.putText(extended_image, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return extended_image

