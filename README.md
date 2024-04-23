# Automatic Decision Shaping
Automatic decision shaping (simulation 1). A system of thermal printers voting to decide which pattern to print.

![screenshot of the simulation](https://raw.githubusercontent.com/olivain/AutomaticDecisionShaping/main/Screenshot%20from%202024-04-23%2015-03-02.png)

# Agent Voting Simulation for Thermal Printer Pattern Selection
This program simulates agents voting collectively to decide which pattern the associated thermal printers will print. Each agent is equipped with a neural network model to make decisions based on given image candidates. The winning pattern is then printed by all the thermal printers.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV (`cv2`)

## Installation

1. Clone or download the repository.
2. Install the required Python packages by running:
    ```
    pip install -r requirements.txt
    ```

## Usage

Run the program with the following command:
   ```
    python3 agent_voting_simulation.py [nb_agents_to_setup] [delay_between_each_vote_in_ms] [training_nb_files] [training_nb_epoch] [--http] [--reset]
   ```
- `nb_agents_to_setup`: Number of agents participating in the simulation.
- `delay_between_each_vote_in_ms`: Delay between each vote in milliseconds.
- `training_nb_files`: Number of training files for neural network training.
- `training_nb_epoch`: Number of epochs for neural network training.
- `--http`: (Optional) Enable HTTP output mode for visualization.
- `--reset`: (Optional) Reset and retrain neural network models.

Example:
  ```
    python3 agent_voting_simulation.py 5 10000 1000 10 --http
  ```

## Features

- **Neural Network Training**: Trains neural network models based on given training data.
- **Agent Voting**: Simulates agent decision-making based on trained models.
- **Thermal Printer Integration**: Prints the winning pattern using the associated thermal printer.
- **HTTP Visualization**: Optional mode to visualize the simulation results via HTTP.

