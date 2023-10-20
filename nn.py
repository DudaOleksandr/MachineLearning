from keras.utils import Progbar

from helper_func import tag_dataset
from model import get_model
from prepro import iterate_minibatches
from validation import compute_f1
import dataset
import tensorflow as tf
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

epochs = 70


train_batch, train_batch_len = dataset.get_train_batch()
dev_batch, dev_batch_len = dataset.get_dev_batch()
test_batch, test_batch_len = dataset.get_test_batch()

wordEmbeddings = dataset.get_word_embeddings()
caseEmbeddings = dataset.get_case_embeddings()
char2Idx, label2Idx, idx2Label = dataset.get_idx()

model = get_model(wordEmbeddings.shape[0], wordEmbeddings.shape[1], wordEmbeddings, caseEmbeddings.shape[1],
                  caseEmbeddings.shape[0], caseEmbeddings, len(char2Idx), len(label2Idx))

for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch, epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches(train_batch, train_batch_len)):
        labels, tokens, casing, char = batch
        model.train_on_batch([tokens, casing, char], labels)
        a.update(i)
    a.update(i + 1)
    print(' ')

model.save("models/model.h5")

#   Performance on dev dataset
predLabels, correctLabels = tag_dataset(dev_batch, model)
pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_batch, model)
pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))

f = open("results.txt", "x")
f.write("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
f.write("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
f.close()
