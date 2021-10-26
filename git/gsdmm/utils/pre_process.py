# -*- coding:utf-8 -*
"""
@author: Cilia Cui
@date: 10-26-2021
"""

import numpy as np
import pandas as pd
import re
from tqdm import tqdm

'''
1 Data Washing: 
    a. split Chinese phrases
    b. delete stopwords
    c. leave Chinese chars only
    d. delete words with low frequency
    e. delete words in length of 1
    f. regular expression for split sentence into parts
    g. get id for each label
'''


def split_words_chinese(dataset, addwords=None, cut_type='acc'):
    '''
    :param dataset: list of sentences(include double list issue)
    :param addwords: list of addwords or files
    :return: splited dataset
    '''
    import jieba
    if addwords:
        jieba.load_userdict(addwords)  # jieba could judge file/list type itself
    outputs = []
    for sentence in tqdm(dataset):
        # print(type(sentence))
        if isinstance(sentence, str):
            pass
        elif isinstance(sentence, int):
            sentence = str(sentence)
        elif isinstance(sentence, list) or isinstance(sentence, np.ndarray):
            sentence = ''.join(sentence)
        else:
            raise IndexError
            # sentence = ''  # error, wrong input type

        if cut_type == 'hmm' or cut_type == 'HMM':
            outputs.append(list(jieba.cut(sentence=sentence, HMM=True)))
        elif cut_type == 'all':
            outputs.append(list(jieba.cut(sentence, cut_all=True)))
        else:
            outputs.append(list(jieba.cut(sentence=sentence, HMM=False)))
    return outputs


def delete_stopwords(dataset, stopwords):
    '''
    :param dataset: double list, like [['1','2','3',...],...]
    :param stopwords: wordlist only
    :return: double list without stopwords
    '''
    outputs = [[x for x in sentence if x not in stopwords] for sentence in tqdm(dataset)]
    return outputs


def chinese_only(dataset):
    '''
    :param dataset: double list, like [['1','2','3',...],...]
    :return: double list with chinese characters only
    '''
    p = re.compile('[\u4e00-\u9fa5]')
    outputs = [[''.join(re.findall(p, x)) for x in sentence] for sentence in dataset]
    return [[x for x in sentence if x not in ('', ' ')] for sentence in outputs]


def del_low_frequence(dataset, min_freq=1):
    '''
    delete the word in dataset when its frequency is lower than the min_freq
    :param min_freq: int, del frequency(words) < min_freq
    :param dataset: double list, like [['1','2','3',...],...]
    :return:
    '''
    from nltk import FreqDist
    # dataset = np.reshape(dataset,(1,-1))[0]
    # print(dataset)
    re_dataset = []
    for x in dataset:
        re_dataset += x
    data_freq = FreqDist(re_dataset)
    del_words = [item for item in tqdm(data_freq) if data_freq[item] < min_freq]
    return delete_stopwords(dataset=dataset, stopwords=del_words)



def del_one_char(dataset):
    '''
    delete the word whose length is 1. for instance, '元', '费', etc.
    :param dataset: double list [[word1,word2,...],...]
    :return: double list without single words.
    '''
    return [[x for x in sen if x.__len__() > 1] for sen in tqdm(dataset)]


def split_parts(dataset, split_word, if_multi=False):
    '''
    :param dataset: list, like ['sentence 1','sentence 2',...]
    :param split_word: the clue that split a sentence into 2 parts
    :return: 2 output lists
    '''
    opt1, opt2 = [], []
    for sentence in dataset:
        if isinstance(sentence, str):
            pass
        elif isinstance(sentence, int):
            sentence = str(sentence)
        else:
            sentence = ''  # error, wrong input type

        opt = sentence.split(split_word)
        if opt.__len__() == 2:
            # print('a')
            opt1.append(opt[0])
            opt2.append(opt[1])
        elif opt.__len__() < 2:
            # print('b')
            opt1.append(opt[0])
            opt2.append('')
        else:
            # print('else')
            if if_multi == 'tail':  # 截尾
                opt1.append(opt[0:-1])
                opt2.append(opt[-1])
            elif if_multi == 'head':  # 截首
                opt1.append(opt[0])
                opt2.append(''.join(opt[1:]))
            else:
                opt1.append(opt[0])
                opt2.append(opt[1])
                print('input error: the split_word has several splits in the sentence:\t{}'.format(sentence))

    return opt1, opt2


def get_vocab_labels(train_labels, save_path, if_save):
    '''
    get ids of the words, one-on-one map
    :param train_labels: list or double list. support [label1, label2,...] type and [[word1, word2,...],[word3,word2,...],...] type
    :param save_path: the path to save the label map in.
    :param if_save: if not, return the map only
    :return: unique map
    '''

    if isinstance(train_labels[0], str):
        all_labels = np.unique(train_labels)
    elif isinstance(train_labels[0], list):
        all_labels = []
        for i, doc in enumerate(train_labels):
            all_labels += doc
        all_labels = np.unique(all_labels)
    else:
        raise TypeError

    if if_save:
        with open(save_path, 'w+', encoding='utf-8') as f:
            for id, value in enumerate(all_labels):
                line = value + '\t' + id.__str__() + '\n'
                f.write(line)
        f.close()
    return all_labels


def mark_df_labels(dataset, input_col, labels_df):
    opt_dataset = pd.merge(dataset, labels_df, on=input_col, how='left')
    return opt_dataset


'''
2 feature: 
    a. TF-IDF & count frequency matrix:
        *use gensim
        sklearn.feature_extraction.text.CountVectorizer
    b. word2vec:
        *use gensim
'''


def create_tfidf(dataset, stopwords=None, type='count', max_feature=10000, asmatrix=False):
    '''
    :param dataset: list of data ['word1 word2','word3 word4',...]
    :param stopwords: list of words ['stop1','stop2',...]
    :param type: choose type, word frequency or tfidf value
    :return: matrix, wordlist, indexlist
    '''
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    if type == 'count':
        # 词袋model
        if stopwords:
            count_vec = CountVectorizer(stop_words=stopwords, encoding='utf-8',
                                        max_features=max_feature)
        else:
            count_vec = CountVectorizer(encoding='utf-8',
                                        max_features=max_feature)


        # HERE: Learn a vocabulary dictionary of all tokens in the raw documents.
        # eg. {'dog':0,'cat':1,'human':2}
        count_vec.fit(dataset)
        count_vec_matrix = count_vec.transform(dataset)
        # count_vec_matrix = count_vec.fit_transform(dataset)
        count_train = np.asarray(count_vec_matrix)
        word_list = count_vec.get_feature_names()
        vocabulary_dict = count_vec.vocabulary_
        return count_train, word_list, vocabulary_dict
    elif type == 'tf':
        # tfidf model countvec + tfidf function
        if stopwords:
            tf_vec = TfidfVectorizer(max_features=max_feature,
                                     stop_words=stopwords, encoding='utf-8')
        else:
            tf_vec = TfidfVectorizer(max_features=max_feature,
                                     encoding='utf-8')
        tf_vec_matrix = tf_vec.fit_transform(dataset)
        print(tf_vec_matrix.shape)
        if asmatrix:
            tf_train = tf_vec_matrix
        else:
            tf_train = np.asarray(tf_vec_matrix)
        word_list = tf_vec.get_feature_names()
        vocabulary_dict = tf_vec.vocabulary_
        return tf_train, word_list, vocabulary_dict

from gensim.models import Word2Vec
import time
def create_word2vec(dataset, size=None, min_count=None, workers=None, save_p=None):
    '''
    see the tutorial: https://rare-technologies.com/word2vec-tutorial/
    :param dataset: cut and filtered double word list
    :return: word2vec model[in useful format]
    '''

    # size: dimension of vector
    # min_count: if frequency < min_count, omit such word
    # workers: multiprocessing, cpu workers
    model = Word2Vec(dataset, size=size, min_count=min_count, workers=workers)


    # accuracy, set paths on your own
    # model.accuracy('/path/testfile')
    # load: model = Word2Vec.load('/path/')
    if save_p:
        # model itself
        model.save(save_p + '/' + str(round(time.time())) + '.model')
        # vector resource
        model.wv.save_word2vec_format(save_p + '/' + str(round(time.time())) + '.vector')
    # continue_train:
    #     model = Word2Vec.load('/path/')
    #     model.train(more_sentences)
    #     model.save(topath)
    return model
