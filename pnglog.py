import numpy as np
import cv2
import random
import os
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading

def stack_images_on_top(existing_file, new_image):
    existing_image = None
    if os.path.exists(existing_file):
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

def crop_image_from_top(image_path,desired_height):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height > desired_height:
        cropped_image = image[0:desired_height, 0:width]
        cv2.imwrite(image_path, cropped_image)
        return True
    return False

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

def start_server():
    server_port = 8000
    server_address = ('', server_port) 
    httpd = HTTPServer(server_address, RequestHandler)
    print(f'Server running on port {server_port}...')
    httpd.serve_forever()

def initializeHttpOutput(nb):
    generate_html(nb, sys.argv[2])
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

def generate_html(png_files_count, reload_time):
    html_content = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Automatic Decision Shaping (simulation 1)</title><style>body,html{background-color:gray;height:100%;margin:0;display:flex;flex-direction:column;justify-content:center}.container{display:flex;flex-direction:column;align-items:center;width:100%}.images{display:flex;flex-direction:row}.boxes{display:flex;justify-content:center}.image{width:10vw;margin-left:calc(2.5vw + 10px);margin-right:calc(2.5vw + 10px);box-shadow:8px 8px 16px rgb(0 0 0 / .4)}.box{width:15vw;height:50px;background-color:#000;height:170px;margin:10px;box-shadow:8px 8px 16px rgb(0 0 0 / .4)}.container img:hover{transform:translateY(10px)}</style></head><body><div class="container"><div class="boxes">"""
    for i in range(png_files_count):
        html_content += "<div class=\"box\"></div>"
    html_content += """</div></div><div class="container" style="height: 100%; margin-top: -80px;"><div class="images">"""
    for i in range(png_files_count):
        r = random.randint(99,99999)
        im = f"columns/{i}.png?t={r}"
        html_content += f" <img src='{im}' class='image' alt='Image {i}'>"
    html_content += f"</div></div><script>document.addEventListener('DOMContentLoaded',function(){{console.log('Page loaded');setInterval(reloadImages,{reload_time})}});"
    html_content += """function reloadImages(){const images=document.querySelectorAll('.image');images.forEach(image=>{const timestamp=new Date().getTime();const src=image.src;image.src=src.includes('?')?src.split('?')[0]+'?'+timestamp:src+'?t='+timestamp;console.log("Reloading image:",image.src)})}</script></body></html>"""
    with open("index.html", "w") as file:
        file.write(html_content)
