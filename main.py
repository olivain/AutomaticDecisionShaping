import tensorflow as tf
import cv2
import random
import utils
import os
import neural
import candidaters
import pnglog
# import printer
import sys
import time
import signal
import sys

def signal_handler(sig, frame):
    print("\nexit ADS...")
    if "--http" in sys.argv:
        os.remove("index.html")
    sys.exit(0)

def main():
    http_output_mode = "--http" in sys.argv
    do_reset = "--reset" in sys.argv

    if len(sys.argv) < 5 or not sys.argv[1].isdigit() or not sys.argv[2].isdigit():
        print(f"\nUSAGE:\npython3 {sys.argv[0]} [nb_agents_to_setup] [delay_between_each_vote_in_ms30000] [training_nb_files] [traning_nb_epoch] ( [--http] [--reset] )\n"
              f"\nExample: python3 {sys.argv[0]} 5 10000 1000 10 --http\n")
        return
  
    signal.signal(signal.SIGINT, signal_handler)

    previous_file = "last_elected_uuid.log"

    delay_votes = int(sys.argv[2])
    nb_models = int(sys.argv[1])
    nb_epoch = int(sys.argv[4])
    nb_dataset = int(sys.argv[3])
    models = []
    type_determinant = random.randint(1, 3)

    models_exist = all(os.path.exists(f"models/agent_{_f}.keras") for _f in range(nb_models))

    directories = ["models", "columns", "candidates", "print"]

    if http_output_mode == True:
        
        pnglog.initializeHttpOutput(nb_models)
        
        files = os.listdir("columns")
        for file in files:
            file_path = os.path.join("columns", file)
            os.remove(file_path)
            print(f". {file_path} deleted.")
                
    if not models_exist or do_reset:
        for directory in directories:
            if os.path.exists(directory) and os.path.isdir(directory):
                files = os.listdir(directory)
                for file in files:
                    file_path = os.path.join(directory, file)
                    os.remove(file_path)
                    print(f". {file_path} deleted.")
            else:
                os.makedirs(directory)
                print(f"New data directory '{directory}' created.")

            if os.path.exists(previous_file):
                    os.remove(previous_file)
        
        models = neural.train(nb_models,type_determinant, nb_epoch, nb_dataset)

        for idx, m in enumerate(models):
            print(f"saving model number {idx} : models/agent_{idx}.keras")
            m.save(f"models/agent_{idx}.keras")
        print("")
        utils.keep_three_pairs_random("candidates")
    else:

        models = [tf.keras.models.load_model(f"models/agent_{_f}.keras") for _f in range(nb_models)]
        print("\nDONE LOADING MODELS")

        if http_output_mode and os.path.exists("columns") and os.path.isdir("columns"):
            files = os.listdir("columns")
            for file in files:
                file_path = os.path.join("columns", file)
                os.remove(file_path)
                print(f". {file_path} deleted.")
  
    agent_tendancies_desc = [
        "(vote for the shape with the most sides)",
        "(vote for the historically least elected)",
        "(vote for the blackest image)",
        f"(vote for the most symmetrical shape)",
        "(vote for the biggest fractal dimension) ",
        "(vote for the most uncommon shapes)",
        "(vote for a certain type of shape)",
        "(vote for the closest to golden ratio)",
        "(vote for the highest aspect ratio)",
        ""        
    ]

    previously_elected = ""

    try:
        with open(previous_file, 'r') as file:
            previously_elected = file.read()
    except FileNotFoundError:
        print("Previously elected UUID not found.")
        previously_elected = ""
        
    while True:

        _fuuid, _guuid = "", ""
        while _fuuid == _guuid:
            if random.choice([True, False]):
                print("Picking an existing UUID for candidate")
                image2, json2 = utils.load_random_candidate("candidates")
            else:
                print("Generating a new candidate")
                json2, image2 = candidaters.generate_new_candidate()
            _fuuid = json2["uuid"]

            if previously_elected:
                filename = f"candidates/{previously_elected}.json"
                image1, json1 = utils.load_candidate_from_file(filename)
                print("Previously elected UUID found :", json1["uuid"])
                if not os.path.exists(filename):
                    json1, image1 = candidaters.generate_new_candidate()
            else:
                print("No previously elected UUID found. Choosing random candidate.")
                image1, json1 = utils.load_random_candidate("candidates")
            _guuid = json1["uuid"]

        elected = 0
        text_print = []

        while elected == 0:
            res_1, res_2 = 0,0
            for i, model in enumerate(models):
                print(f"----\n  [+] Agent number {i}:")
                if i < len(agent_tendancies_desc):
                    print("     ", agent_tendancies_desc[i])

                decision_agent1, decision_agent2 = neural.agent_decision(model, image1, json1, image2, json2)

                print(f"       --Decision Image 1: {decision_agent1}")
                print(f"       --Decision Image 2: {decision_agent2}")

                if decision_agent1 > decision_agent2:
                    print("     Voted for image 1.")
                    text_print.append("VOTED NO FOR THE CHANGE")
                    res_1 += 1
                else:
                    print("     Voted for image 2.")
                    text_print.append("VOTED YES FOR THE CHANGE")
                    res_2 += 1
            print("Vote is over")

            if res_1 > res_2:
                elected_json = json1
                victory = image1
                elected = 1
            elif res_2 > res_1:
                elected_json = json2
                victory = image2
                elected = 2

            if elected == 0:
                print(f"     yes:{res_2}/no:{res_1} = status quo ! let's revote !!!!\n")

        elected_json["nb_elected"] += 1
        json_file_name = f"candidates/{elected_json['uuid']}.json"
        utils.write_candidate_to_json(elected_json, json_file_name)
        y_pos = elected_json["box"][1] + elected_json["box"][3]  # y + height
        previously_elected = elected_json["uuid"]

        print(f"     [!] IMAGE NUMBER {elected} IS ELECTED ! \n")
        
        with open(previous_file, 'w') as file:
            file.write(previously_elected)

        # Perform decision making
        for i in range(nb_models):
            img_with_text = candidaters.create_text_image_underneath(victory, text_print[i],y_pos)
            cv2.imwrite(f"print/{i}.png",img_with_text)
            
            if http_output_mode == True:
                http_file = f"columns/{i}.png"
                pnglog.stack_images_on_top(http_file, img_with_text)
                pnglog.crop_image_from_top(http_file,5000)              

        # send to the thermal printer (POS 5890)
        #printer.print_images(nb_models) # TODO ! 

        print("waiting for next election...")
        time.sleep(delay_votes / 1000)

if __name__ == "__main__":
    main()
