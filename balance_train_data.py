from __future__ import print_function
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from model.model import model_selector
from reader.filereader import read_glove_vectors, read_input_data, train_data
from utils import argumentparser

np.random.seed(42)

def main():
    args = argumentparser.ArgumentParser()
    train(args)

def train(args):
    print('Reading word vectors.')
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    print('Found {} word vectors.'.format(len(embeddings_index)))

    print('Processing input data')
    texts, labels = read_input_data(args.data_dir)
    x_train, y_train = train_data(texts, labels)
    # texts - list of text samples
    # labels_index - dictionary mapping label name to numeric id
    # labels - list of label ids
    print('Found {} texts.'.format(len(texts)))

    # Vectorize the text sample into 2D integer tensor
    tokenizer = Tokenizer(nb_words=args.nb_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    tokenizer_train = Tokenizer(nb_words=args.nb_words)
    tokenizer_train.fit_on_texts(x_train)
    sequences_train = tokenizer_train.texts_to_sequences(x_train)

    '''
    data = pad_sequences(sequences, maxlen=args.max_sequence_len)
    my_label = labels
    labels = to_categorical(np.asarray(labels))

    # split the input data into training set and validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    #nb_validation_samples = int(args.validation_split * data.shape[0])
    #x_train = data[:-nb_validation_samples]
    #y_train = labels[:-nb_validation_samples]
    #x_val = data[-nb_validation_samples:]
    #y_val = labels[-nb_validation_samples:]
    print('Loading data.........')
    x_train, y_train = train_data(data, my_label)
    x_test = data

    # Transform labels to be categorical variables

    print('Shape of train data tensor:', x_train.shape)
    print('Shape of train label tensor:', y_train.shape)
    print('Shape of test data tensor:', x_test.shape)
    '''
    '''
    x_train, y_train = train_data(data, labels)
    x_train = pad_sequences(sequences, maxlen=args.max_sequence_len)
    y_train = to_categorical(np.asarray(y_train))
    labels = to_categorical(np.asarray(labels))
    indices = np.arange(x_train)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_val = data
    y_val = labels
    '''
    # Transform labels to be categorical variables

    print('Found total {} unique tokens.'.format(len(word_index)))

    data = pad_sequences(sequences, maxlen=args.max_sequence_len)
    x_train = pad_sequences(sequences_train, maxlen=args.max_sequence_len)

    # Transform labels to be categorical variables
    labels = to_categorical(np.asarray(labels))
    y_train = to_categorical(np.asarray(y_train))
    print('Shape of  train data tensor:', x_train.shape)
    print('Shape of train label tensor:', y_train.shape)
    print('Shape of  test data tensor:', data.shape)
    print('Shape of test label tensor:', labels.shape)

    # split the input data into training set and validation set
    #indices = np.arange(data.shape[0])
    #np.random.shuffle(indices)
    #data_ = data[indices]
    #labels_ = labels[indices]
    #nb_validation_samples = int(args.validation_split * data.shape[0])

    #x_train = data_[:-nb_validation_samples]
    #y_train = labels_[:-nb_validation_samples]
    #x_val = data_[-nb_validation_samples:]
    #y_val = labels_[-nb_validation_samples:]

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data_ = data[indices]
    labels_ = labels[indices]

    print('Preparing embedding matrix.')

    # initiate embedding matrix with zero vectors.
    nb_words = min(args.nb_words, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    args.nb_words = nb_words
    args.len_labels_index = 3

    model = model_selector(args, embedding_matrix)

    checkpoint_filepath = os.path.join(args.model_dir, "new.en.msd.weights.best.hdf5")
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    model_json = model.to_json()
    with open(os.path.join(args.model_dir, "new.en.msd.model.json"), "w") as json_file:
        json_file.write(model_json)

    model.fit(x_train, y_train, validation_data=(data_, labels_), nb_epoch=args.num_epochs, batch_size=args.batch_size, callbacks=callbacks_list, verbose=1)
    proba = model.predict_proba(data, batch_size=300)
    np.savetxt('new_en_msd', proba, delimiter='\t', fmt='%.6f')


if __name__ == '__main__':
    main()
