from __future__ import print_function
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from reader.filereader import read_glove_vectors, read_input_data
from utils import argumentparser

np.random.seed(42)

def main():
    args = argumentparser.ArgumentParser()
    train(args)

def score_list(file):
    f = open(file, 'r')
    mono = []
    swap = []
    disc = []

    lines = f.readlines()
    for line in lines:
        line = line.strip().decode('utf-8').split()
        mono.append(float(line[0]))
        swap.append(float(line[1]))
        disc.append(float(line[2]))
        return mono, swap, disc

def train(args):
    print('loading weight matrix .....................')
    model = load_model('model/renew.en.msd.weights.best.hdf5')

    #print('Reading word vectors.')
    #embeddings_index = read_glove_vectors(args.embedding_file_path)
    #print('Found {} word vectors.'.format(len(embeddings_index)))

    print('Processing input data')
    texts, labels = read_input_data(args.data_dir)
    # texts - list of text samples
    # labels_index - dictionary mapping label name to numeric id
    # labels - list of label ids
    print('Found {} texts.'.format(len(texts)))

    # Vectorize the text sample into 2D integer tensor
    tokenizer = Tokenizer(nb_words=args.nb_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    data = pad_sequences(sequences, maxlen=args.max_sequence_len)

    df = pd.DataFrame(data.tolist())
    mono, swap, disc = score_list('data/en_msd_score')
    df.columns = ['data']
    df['mono'] = mono
    df['swap'] = swap
    df['disc'] = disc

    rules = df.groupby(['data'], as_index=False).mean()
    df.to_csv('dataframe', sep='\t', encoding='utf-8', index=False)
    print(rules)

    #for i in range(100):
        #print(i, data[i+200])
    #print(data[223])
    #print(data[225])
    #proba_1 = model.predict_proba(data[223])
    #proba_2 = model.predict_proba(data[225])
    #print(proba_1)
    #print(proba_2)
    #np.savetxt('renew_cnn_en_msd', proba, delimiter='\t', fmt='%.6f')

if __name__ == '__main__':
    main()
