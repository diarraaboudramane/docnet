# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:18:26 2025

@author: BlueSky
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, output_dir, epoch, is_train=True):
    """
    This function plots and saves the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Epoch {epoch} - {"Train" if is_train else "Validation"}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'conf_matrix_epoch_{epoch}_{"train" if is_train else "val"}.png')
    plt.savefig(output_path)
    plt.show()