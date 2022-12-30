import os
import string
import re
from collections import Counter
from itertools import groupby
from tqdm import tqdm
from string import punctuation
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def enumrate_files_by_group(data_dir: str):
    files_list = []
    for dirs, _, files in os.walk(data_dir):
        for file in files:
            files_list.append(os.path.join(dirs, file))
    
    files_list.sort(key=lambda x: x.split(os.sep)[-2])
    
    group_to_files = {}
    for keys, group in groupby(files_list, key=lambda x: x.split(os.sep)[-2]):
        group_to_files[keys] = list(group)
        
    return group_to_files

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

def remove_digits(text):
    return re.sub(r'[~^0-9]', '', text)


def compute_tfidf(files_to_path, files_to_content, vocab, vocab_to_index):
    termsFrequency = {}
    documentFrequency = {}
    N = len(files_to_content)
    assert N == len(files_to_path)
    
    for doc in files_to_content:
        content = files_to_content[doc]
        frequency = dict(Counter(content))
        termsFrequency[doc] = frequency
        for word in frequency:
            if word in documentFrequency:
                documentFrequency[word] += 1
            else:
                documentFrequency[word] = 1
                
    #print(termsFrequency)
    #print(documentFrequency)
    
    tfidf_vector_table = {}
    tfidf_array = []
    for doc in termsFrequency:
        tfidf_vector = np.zeros(shape=[len(vocab)], dtype=np.float32)
        frequency = termsFrequency[doc]
        for word in frequency:
            if word in vocab:
                tf = frequency[word]
                df = documentFrequency[word]
                tfidf = tf * np.log(N / df)
                tfidf_vector[vocab_to_index[word]] = tfidf
            
        tfidf_vector_table[doc] = tfidf_vector
        tfidf_array.append(tfidf_vector)
    
    tfidf_array = np.stack(tfidf_array, axis=0)

if __name__ == "__main__":

    data_dir = r"20news-bydate"
    train_dir = os.path.join(data_dir, "20news-bydate-{}".format("train"))
    test_dir = os.path.join(data_dir, "20news-bydate-{}".format("test"))

    train_group_to_files = enumrate_files_by_group(data_dir=train_dir)

    train_files_to_path = {}

    for group in train_group_to_files:
        file_list = train_group_to_files[group]
        for file in file_list:
            train_files_to_path[os.path.join(group, os.path.basename(file))] = file

    test_group_to_files = enumrate_files_by_group(data_dir=test_dir)

    test_files_to_path = {}

    for group in test_group_to_files:
        file_list = test_group_to_files[group]
        for file in file_list:
            test_files_to_path[os.path.join(group, os.path.basename(file))] = file

    train_files_to_content = {}

    for file in tqdm(train_files_to_path):
        with open(train_files_to_path[file], "r") as f:
            content = f.read()

        train_files_to_content[file] = content

    print(train_files_to_content['talk.religion.misc\\83728'])

    test_files_to_content = {}

    for file in tqdm(test_files_to_path):
        with open(test_files_to_path[file], "r") as f:
            content = f.read()

        test_files_to_content[file] = content

    with open(os.path.join(data_dir, "vocabulary.txt"), "r") as f:
        vocab = f.read()
        vocab = vocab.split("\n")
        vocab = list(filter(None, vocab))
    #print(vocab, len(vocab))
    
    index_to_vocab = dict(zip(list(range(len(vocab))), vocab))
    vocab_to_index = {v: k for k, v in index_to_vocab.items()}

    # Remove punctuation
    train_files_to_content = {k: remove_punctuation(v) for k, v in train_files_to_content.items()}
    test_files_to_content = {k: remove_punctuation(v) for k, v in test_files_to_content.items()}

    print(train_files_to_content['talk.religion.misc\\83728'])

    # Lower
    train_files_to_content = {k: v.lower() for k, v in train_files_to_content.items()}
    test_files_to_content = {k: v.lower() for k, v in test_files_to_content.items()}

    print(train_files_to_content['talk.religion.misc\\83728'])

    # Remove digits
    train_files_to_content = {k: remove_digits(v) for k, v in train_files_to_content.items()}
    test_files_to_content = {k: remove_digits(v) for k, v in test_files_to_content.items()}

    print(train_files_to_content['talk.religion.misc\\83728'])

    # Word Tokenization
    train_files_to_content = {k: word_tokenize(v) for k, v in train_files_to_content.items()}
    test_files_to_content = {k: word_tokenize(v) for k, v in test_files_to_content.items()}

    print(train_files_to_content['talk.religion.misc\\83728'])
    
    train_index_to_files_name = dict(zip(range(len(train_files_to_path.keys())), train_files_to_path.keys()))
    test_index_to_files_name = dict(zip(range(len(test_files_to_path.keys())), test_files_to_path.keys()))
    
    group_to_index = dict(zip(train_group_to_files.keys(), range(len(train_group_to_files.keys()))))
    
    train_file_name_to_group = {}
    for group in train_group_to_files:
        for file in train_group_to_files[group]:
            train_file_name_to_group[file] = group
            
    test_file_name_to_group = {}
    for group in test_group_to_files:
        for file in test_group_to_files[group]:
            test_file_name_to_group[file] = group
    
    print(group_to_index)
    train_groundtruth = [group_to_index[train_file_name_to_group[x]] for x in train_files_to_path.values()]
    test_groundtruth = [group_to_index[test_file_name_to_group[x]] for x in test_files_to_path.values()]
    print(len(train_groundtruth), len(test_groundtruth))
    
    train_documents = [v for k, v in train_files_to_content.items()]
    test_documents = [v for k, v in test_files_to_content.items()]
    
    #print(train_documents[:10], test_documents[:10])
    
    tfidf = TfidfVectorizer(tokenizer=lambda x: x,
                        preprocessor=lambda x: x, stop_words='english', vocabulary=vocab_to_index)
    train_vectors = tfidf.fit_transform(train_documents)
    
    print(train_vectors.shape)
    
    test_vectors = tfidf.fit_transform(test_documents)
    
    print(test_vectors.shape)
    
    clustering = KMeans(n_clusters=20, init='k-means++', max_iter=100, random_state=100)
    #clustering = cluster.BisectingKMeans(n_clusters=20, init='k-means++', max_iter=100, random_state=100)
    clustering.fit(train_vectors)
    
    print(clustering.labels_)
    
    predict_test = clustering.predict(test_vectors)
    #predict_test = clustering.fit_predict(test_vectors)
    print(predict_test)
    
    print("Train Homogeneity: {}".format(metrics.homogeneity_score(train_groundtruth, clustering.labels_)))
    print("Train Completeness: {}".format(metrics.completeness_score(train_groundtruth, clustering.labels_)))
    print("Train V-measure: {}".format(metrics.v_measure_score(train_groundtruth, clustering.labels_)))
    print("Train Adjusted Rand-Index: {}".format(metrics.adjusted_rand_score(train_groundtruth, clustering.labels_)))
    print("Train Silhouette Coefficient: {}".format(metrics.silhouette_score(train_vectors, clustering.labels_, sample_size=5000)))
    
    print("Test Homogeneity: {}".format(metrics.homogeneity_score(test_groundtruth, predict_test)))
    print("Test Completeness: {}".format(metrics.completeness_score(test_groundtruth, predict_test)))
    print("Test V-measure: {}".format(metrics.v_measure_score(test_groundtruth, predict_test)))
    print("Test Adjusted Rand-Index: {}".format(metrics.adjusted_rand_score(test_groundtruth, predict_test)))
    print("Test Silhouette Coefficient: {}".format(metrics.silhouette_score(test_vectors, predict_test, sample_size=5000)))
    
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    
    lsa_train_vectors = lsa.fit_transform(train_vectors)
    lsa_test_vectors = lsa.fit_transform(test_vectors)
    
    print(lsa_train_vectors.shape)
    print(lsa_test_vectors.shape)
    
    clustering = KMeans(n_clusters=20, init='k-means++', max_iter=100, random_state=100)
    #clustering = cluster.BisectingKMeans(n_clusters=20, init='k-means++', max_iter=100, random_state=100)
    clustering.fit(lsa_train_vectors)
    
    print(clustering.labels_)
    
    predict_test = clustering.predict(lsa_test_vectors)
    #predict_test = clustering.fit_predict(test_vectors)
    print(predict_test)
    
    print("LSA Train Homogeneity: {}".format(metrics.homogeneity_score(train_groundtruth, clustering.labels_)))
    print("LSA Train Completeness: {}".format(metrics.completeness_score(train_groundtruth, clustering.labels_)))
    print("LSA Train V-measure: {}".format(metrics.v_measure_score(train_groundtruth, clustering.labels_)))
    print("LSA Train Adjusted Rand-Index: {}".format(metrics.adjusted_rand_score(train_groundtruth, clustering.labels_)))
    print("LSA Train Silhouette Coefficient: {}".format(metrics.silhouette_score(lsa_train_vectors, clustering.labels_, sample_size=5000)))
    
    print("LSA Test Homogeneity: {}".format(metrics.homogeneity_score(test_groundtruth, predict_test)))
    print("LSA Test Completeness: {}".format(metrics.completeness_score(test_groundtruth, predict_test)))
    print("LSA Test V-measure: {}".format(metrics.v_measure_score(test_groundtruth, predict_test)))
    print("LSA Test Adjusted Rand-Index: {}".format(metrics.adjusted_rand_score(test_groundtruth, predict_test)))
    print("LSA Test Silhouette Coefficient: {}".format(metrics.silhouette_score(lsa_test_vectors, predict_test, sample_size=5000)))
    
    lsa_vectorizer = make_pipeline(
    HashingVectorizer(stop_words="english", n_features=50_000, tokenizer=lambda x: x, preprocessor=lambda x: x),
    TfidfTransformer(),
    TruncatedSVD(n_components=100, random_state=0),
    Normalizer(copy=False),
    )
    
    hashed_lsa_train_vectors = lsa_vectorizer.fit_transform(train_documents)
    
    print(hashed_lsa_train_vectors.shape)
    
    hashed_lsa_test_vectors = lsa_vectorizer.fit_transform(test_documents)
    
    print(hashed_lsa_test_vectors.shape)
    
    clustering = KMeans(n_clusters=20, init='k-means++', max_iter=100, random_state=100)
    #clustering = cluster.BisectingKMeans(n_clusters=20, init='k-means++', max_iter=100, random_state=100)
    clustering.fit(hashed_lsa_train_vectors)
    
    print(clustering.labels_)
    
    predict_test = clustering.predict(hashed_lsa_test_vectors)
    #predict_test = clustering.fit_predict(test_vectors)
    print(predict_test)
    
    print("Hased LSA Train Homogeneity: {}".format(metrics.homogeneity_score(train_groundtruth, clustering.labels_)))
    print("Hased LSA Train Completeness: {}".format(metrics.completeness_score(train_groundtruth, clustering.labels_)))
    print("Hased LSA Train V-measure: {}".format(metrics.v_measure_score(train_groundtruth, clustering.labels_)))
    print("Hased LSA Train Adjusted Rand-Index: {}".format(metrics.adjusted_rand_score(train_groundtruth, clustering.labels_)))
    print("Hased LSA Train Silhouette Coefficient: {}".format(metrics.silhouette_score(hashed_lsa_train_vectors, clustering.labels_, sample_size=5000)))
    
    print("Hashed LSA Test Homogeneity: {}".format(metrics.homogeneity_score(test_groundtruth, predict_test)))
    print("Hashed LSA Test Completeness: {}".format(metrics.completeness_score(test_groundtruth, predict_test)))
    print("Hashed LSA Test V-measure: {}".format(metrics.v_measure_score(test_groundtruth, predict_test)))
    print("Hashed LSA Test Adjusted Rand-Index: {}".format(metrics.adjusted_rand_score(test_groundtruth, predict_test)))
    print("Hashed LSA Test Silhouette Coefficient: {}".format(metrics.silhouette_score(hashed_lsa_test_vectors, predict_test, sample_size=5000)))
    
    '''# TfIdf
    
    train_termsFrequency = {}
    train_documentFrequency = {}
    N = len(train_files_to_content)
    assert N == len(train_files_to_path)
    
    for doc in train_files_to_content:
        content = train_files_to_content[doc]
        frequency = dict(Counter(content))
        train_termsFrequency[doc] = frequency
        for word in frequency:
            if word in train_documentFrequency:
                train_documentFrequency[word] += 1
            else:
                train_documentFrequency[word] = 1
                
    #print(train_termsFrequency)
    #print(train_documentFrequency)
    
    train_tfidf_vector_table = {}
    train_tfidf_array = []
    for doc in train_termsFrequency:
        tfidf_vector = np.zeros(shape=[len(vocab)], dtype=np.float32)
        frequency = train_termsFrequency[doc]
        for word in frequency:
            if word in vocab:
                tf = frequency[word]
                df = train_documentFrequency[word]
                tfidf = tf * np.log(N / df)
                tfidf_vector[vocab_to_index[word]] = tfidf
            
        train_tfidf_vector_table[doc] = tfidf_vector
        train_tfidf_array.append(tfidf_vector)
    
    train_tfidf_array = np.stack(train_tfidf_array, axis=0)
    print(len(train_tfidf_vector_table), train_tfidf_array.shape)
    
    test_termsFrequency = {}
    test_documentFrequency = {}
    N = len(test_files_to_content)
    assert N == len(test_files_to_path)
    
    for doc in test_files_to_content:
        content = test_files_to_content[doc]
        frequency = dict(Counter(content))
        test_termsFrequency[doc] = frequency
        for word in frequency:
            if word in test_documentFrequency:
                test_documentFrequency[word] += 1
            else:
                test_documentFrequency[word] = 1
                
    #print(test_termsFrequency)
    #print(test_documentFrequency)
    
    test_tfidf_vector_table = {}
    test_tfidf_array = []
    for doc in test_termsFrequency:
        tfidf_vector = np.zeros(shape=[len(vocab)], dtype=np.float32)
        frequency = test_termsFrequency[doc]
        for word in frequency:
            if word in vocab:
                tf = frequency[word]
                df = test_documentFrequency[word]
                tfidf = tf * np.log(N / df)
                tfidf_vector[vocab_to_index[word]] = tfidf
            
        test_tfidf_vector_table[doc] = tfidf_vector
        test_tfidf_array.append(tfidf_vector)
    
    test_tfidf_array = np.stack(test_tfidf_array, axis=0)
    print(len(test_tfidf_vector_table), test_tfidf_array.shape)
    
    train_tfidf_vector_table, train_tfidf_array = compute_tfidf(files_to_path=train_files_to_path, files_to_content=train_files_to_content, vocab=vocab, vocab_to_index=vocab_to_index)
    print(len(train_tfidf_vector_table), train_tfidf_array.shape)
    
    test_tfidf_vector_table, test_tfidf_array = compute_tfidf(files_to_path=test_files_to_path, files_to_content=test_files_to_content, vocab=vocab, vocab_to_index=vocab_to_index)
    print(len(test_tfidf_vector_table), test_tfidf_array.shape)'''