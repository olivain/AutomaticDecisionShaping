import tensorflow as tf
import cv2
import random
import utils
import os
import neural
import candidaters
import pnglog
import printer
import sys
import time
import signal
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatic Decision Shaping. 2024.")
    parser.add_argument("nb_agents", type=int, help="Number of agents to setup")
    parser.add_argument("delay_between_votes", type=int, help="Delay between each vote in milliseconds")
    parser.add_argument("training_nb_files", nargs="?", type=int, default=None, help="Training : number of files to generate as dataset")
    parser.add_argument("training_nb_epoch", nargs="?", type=int, default=None, help="Training : number of epochs")
    
    parser.add_argument("--reset", action="store_true", help="Reset all datas")
    parser.add_argument("--http", action="store_true", help="Enable HTTP output on localhost:8000")
    args = parser.parse_args()

    # Check if both training_nb_files and training_nb_epoch are specified
    if args.training_nb_files is not None and args.training_nb_epoch is not None:
        args.reset = True
    return args

def signal_handler(sig, frame, http_output_mode):
    print("\nexit ADS...")
    if http_output_mode:
        os.remove("index.html")
    sys.exit(0)

def main():
    print("\n\n🖨️ Automatic decision shaping 🗳️\n")
    args = parse_arguments()

    # Access arguments
    nb_models = args.nb_agents
    delay_votes = args.delay_between_votes
    nb_dataset = args.training_nb_files
    nb_epoch = args.training_nb_epoch

    http_output_mode = args.http
    do_reset = args.reset

    type_determinant = random.randint(1, 3)

    print("nb_agents:", nb_models)
    print("delay_between_votes:", delay_votes)
    print("training_nb_files:", nb_dataset)
    print("training_nb_epoch:", nb_epoch)
    print("http_output_mode:", http_output_mode)
    print("do_reset:", do_reset)
  
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, http_output_mode))

    previous_file = "last_elected_uuid.log"

    models_exist = all(os.path.exists(f"models/agent_{_f}.keras") for _f in range(nb_models))
    directories = ["models", "columns", "candidates", "print"]
    
    print(f"models found:{models_exist}")
    
    if http_output_mode == True:
            pnglog.initializeHttpOutput(nb_models)

    if not models_exist or do_reset:
        if nb_epoch == None or nb_dataset == None:
            print("[!] no epoch or dataset number specified.\n    Default: nb_epoch=10 nb_dataset=1000")
            nb_epoch=10
            nb_dataset=1000

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

        if http_output_mode == True:
            files = os.listdir("columns")
            for file in files:
                file_path = os.path.join("columns", file)
                os.remove(file_path)
                print(f". {file_path} deleted.")
        
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
                print("Generating a new candidate")
                json2, image2 = candidaters.generate_new_candidate()

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
                pnglog.crop_image_from_top(http_file,15000)              

        # send to the thermal printer (POS 5890)
        printer.print_images(nb_models) 

        print("waiting for next election...")
        time.sleep(delay_votes / 1000)

if __name__ == "__main__":
    main()
