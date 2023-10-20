import numpy as np
from keras.utils import Progbar


def tag_dataset(dataset, model):
    correct_labels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        correct_labels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i + 1)
    return predLabels, correct_labels
