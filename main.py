
""" 
Main script.

Argument parsing and such.
"""

import models
import trainers
import argparse


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", type=str, default="viz", help="Whether to train on CIFAR10 or vizualize")
    ap.add_argument("-m", "--model", type=str, default="vgg", help="Model architecture to use")
    ap.add_argument("-d", "--data-root", type=str, default="data/imagenet/train/", help="Images root with folder names as labels")
    ap.add_argument("-l", "--model-ckpt", type=str, help="Path to model checkpoint if not pretrained model")
    ap.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for dataloader")
    ap.add_argument("-e", "--method", type=str, default="channel", help="Channelwise (channel) or locationwise (spatial) ablation")
    ap.add_argument("-f", "--final-activation", type=str, default="relu", help="Whether final activation is relu or softmax")
    ap.add_argument("-c", "--clf-activation", type=str, default="softmax", help="Whether class activation is sigmoid or softmax")
    ap.add_argument("-n", "--num-batches", type=int, default=1, help="Number of batches for which to run")
    args = vars(ap.parse_args())

    if args["task"] == "viz":    
        model = models.AblationCAM(args)
        model.run_for_batches(args["num_batches"])
    
    elif args["task"] == "train":
        trainer = trainers.ModelTrainer(args)
        trainer.train(epochs=10, eval_every=2)

    else:
        raise ValueError(f"Unrecognized task {args['task']}")