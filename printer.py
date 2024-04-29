from escpos import printer 
import utils
import os
# import time

def get_existing_lp_files():
    usb_dir = '/dev/usb'
    lp_files = [os.path.join(usb_dir, file) for file in os.listdir(usb_dir) if file.startswith('lp')]
    return lp_files

def print_images(nb_models):
    try:
        lp_files = get_existing_lp_files()

        for i, path in enumerate(lp_files):
            if i < nb_models:
                try:
                    p = printer.File(devfile=path)
                    image_path = f"print/{i}.png"
                    utils.rotate_image_180(image_path, "tmp.png")
                    p.image('tmp.png')
                except Exception as e:
                    print(f"Error printing image {i}: {e}")
                    continue
        if os.path.exists("tmp.png"):
            os.remove('tmp.png')
            return

    except Exception as e:
                    print(f"Error printing: {e}")
                    return
