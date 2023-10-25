#importing necessary librarys
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_training_results(model_name:str, results_dir = RESULTS_DIR):


    ## load the training loss and accuracy
    train_loss = np.load(os.path.join(results_dir, model_name, 'train_loss.npy'))
    train_accuracy = np.load(os.path.join(results_dir, model_name, 'train_accuracy.npy'))
    ## load the validation loss and accuracy
    val_loss = np.load( os.path.join(results_dir, model_name, 'val_loss.npy'))
    val_accuracy = np.load(os.path.join(results_dir, model_name, 'val_accuracy.npy'))

    ## plot the training loss and accuracy
    plt.figure(figsize=(16,8))
    plt.plot(train_loss,color="r",marker="o", label="Training Loss")
    plt.plot(train_accuracy, color='b', marker='x')

    ## plot the validation loss and accuracy
    plt.plot(val_loss,color="g",marker="o", label="Validation Loss")
    plt.plot(val_accuracy, color='y', marker='x', label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Loss and Accuracy")
    plt.legend(["Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"])
    plt.show()


if __name__ == '__main__':
    plot_training_results(model_name='MobileNetV2')