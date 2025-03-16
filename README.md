# EidolonZero
Repository for the development of the **EidolonZero** bot, which plays the game of fog-of-war chess.

This repository will contain both the training life-cycle, the backend AI model architecture,
and also the interface from which an end user can play against this bot.

## Setting up the Environment

After this repository is `git clone`d,
run the `makeenv.cmd` batch file to create the `.venv` folder that contains the virtual environment.

Then, you can use `runserver.cmd` to start up the Flask web server.
The sample `interface/index.html` will then work as intended when the button is clicked.

## Navigating the File Structure

The `architectures` folder contains the definition of the PyTorch modules
that will be used to implement the backend AI engine of **EidolonZero**,
as well as the tree searching mechanism and the custom loss functions.

The `boards` folder defines the fundamental data types that are used
during the training of the engine as well as the facilitation of its execution in production.

The `evaluation` folder defines the processes used to measure its playing strength
both against its own model snapshots and against a random baseline player.

The `interface` folder is where all the user interface assets are,
including the main page `index.html`.

The `models` folder will store all the recorded snapshots of the model weights.

The `training` folder contains the procedures that
carry out the training pipeline of **EidolonZero**.

**EidolonZero** itself has its functionalities accessible from `app.py`.
