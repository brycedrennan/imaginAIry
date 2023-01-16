# Adding a concept to Stable Diffusion

You can use Imaginairy to teach the model a new concept (a person, thing, style, etc) using the `aimg train-concept`
command. 

## Requirements
 - Graphics card: 3090 or better
 - Linux
 - A working Imaginairy installation
 - a folder of images of the concept you want to teach the model

## Background

To train the model we show it a lot of images of the concept we want to teach it. The problem is the model can easily
overfit to the images we show it. To prevent this we also show it images of the class of thing that is being trained.  
Imaginairy will generate the images needed for this before running the training job.

Provided a directory of concept images, a concept token, and a class token, this command will train the model
to generate images of that concept.


This happens in a 3-step process:

1. Cropping and resizing your training images. If --person is set we crop to include the face.
2. Generating a set of class images to train on. This helps prevent overfitting.
3. Training the model on the concept and class images.

The output of this command is a new model weights file that you can use with the --model option.



## Instructions

1. Gather a set of images of the concept you want to train on. The images should show the subject from a variety of angles
   and in a variety of situations.
2. Run `aimg train-concept` to train the model. 

   - Concept label: For a person, firstnamelastname should be fine.
       - If all the training images are photos you should add "a photo of" to the beginning of the concept label.
   - Class label: This is the category of the things beings trained on. For people this is typically "person", "man"
     or "woman".
       - If all the training images are photos you should add "a photo of" to the beginning of the class label.
       - CLass images will be generated for you if you do not provide them.

    For example, if you were training on photos of a man named bill hamilton you could run the following:
    
    ```
    aimg train-concept \\
        --person \\
        --concept-label "photo of billhamilton man" \\
        --concept-images-dir ./images/billhamilton \\
        --class-label "photo of a man" \\
        --class-images-dir ./images/man
    ```
3. Stop training before it overfits.  
    - The training script will output checkpoint ckpt files into the logs folder of wherever it is run from. You can also
monitor generated images in the logs/images folder. They will be the ones named "sample"
    - I don't have great advice on when to stop training yet. I stopped mine at epoch 62 at it didn't seem quite good enough, at epoch 111 it 
produced my face correctly 50% of the time but also seemed overfit in some ways (always placing me in the same clothes or background as training photos).
    - You can monitor model training progress in Tensorboard. Run `tensorboard --logdir lightning_logs` and open the link it gives you in your browser.

4. Prune the model to bring the size from 11gb to ~4gb: `aimg prune-ckpt logs/2023-01-15T05-52-06/checkpoints/epoch\=000049.ckpt`. Copy it somewhere
and give it a meaninful name.

## Using the new model
You can reference the model like this in imaginairy: 
`imagine --model my-models/billhamilton-man-e111.ckpt`

When you use the model you should prompt with `firstnamelastname classname` (e.g. `billhamilton man`).


## Disclaimers

- The settings imaginairy uses to train the model are different than other software projects. As such you cannot follow
advice you may read from other tutorials regarding learning rate, epochs, steps, batch size.  They are not directly 
comparable. In laymans terms the "steps" are much bigger in imaginairy.
- I consider this training feature experimental and don't currently plan to offer support for it. Any further work will 
be at my leisure. As a result I may close any reported issues related to this feature.
- You can find a lot more relevant information here: https://github.com/JoePenna/Dreambooth-Stable-Diffusion

## Todo
 - figure out how to improve consistency of quality from trained model
 - train on the depth guided model instead of SD 1.5 since that will enable more consistent output
 - figure out metric to use for stopping training
 - possibly swap out and randomize backgrounds on training photos so over-fitting does not occur