from escpos.printer import Usb

def print_images(nb_models):
    #Usb(0x416, 0x5011, in_ep=0x81, out_ep=0x03, profile="POS-5890")
    printer = [
        [0x416, 0x5011, 0x81, 0x03]
    ]
    nb_printer = 0
    for i in range(nb_models):
        f = f"print/{i}.png"
        if printer[nb_printer] != None:
            d = printer[nb_printer]
            p = Usb(d[0], d[1], in_ep=d[2], out_ep=d[3], profile="POS-5890")
            # Print the image
            p.image(f)