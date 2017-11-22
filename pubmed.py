import numpy as np

import pickle
import logging
import string

from Bio import Entrez

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.externals import joblib

EMAIL = ""

SEARCH_TERMS = ["local field potential",
                "neuropharmacology",
                "neuroanatomy",
                "neurophilosophy",
                "computational neuroscience",
                "neurophysiology",
                "functional neuroimaging",
                "neuroimage[Journal]"]

# Edit these values
n_features = 1000
n_topics = 10
n_top_words = 20

def fetch_pubmed_abstracts(search_term):
    Entrez.email = EMAIL

    texts = []
    search_handle = Entrez.esearch(db="pubmed", term=search_term, retmax=100000)
    search_record = Entrez.read(search_handle)

    ids = search_record["IdList"]

    fetch_handle = Entrez.efetch(db="pubmed", id=','.join(ids), retmode="xml")
    fetch_record = Entrez.read(fetch_handle)
    for article in fetch_record['PubmedArticle']:
        try:
            text = str(article['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
            texts.append(text)
        except KeyError:
            logging.info('Invalid key')
            continue

    fetch_handle.close()
    search_handle.close()
    return texts

def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)

    wordnet_lemmatizer = WordNetLemmatizer()

    return wordnet_lemmatizer.lemmatize(token, tag)

def tokenize(document, lower=True, strip=True):
    # Break the document into sentences
    for sent in sent_tokenize(document):
        # Break the sentence into part of speech tagged tokens
        for token, tag in pos_tag(wordpunct_tokenize(sent)):
            # Apply preprocessing to the token
            token = token.lower() if lower else token
            token = token.strip() if strip else token
            token = token.strip('_') if strip else token
            token = token.strip('*') if strip else token

            if token.isnumeric():
                continue

            if len(token) == 1:
                continue

            # If stopword, ignore token and continue
            if token in set(sw.words('english')):
                continue

            # If punctuation, ignore token and continue
            if all(char in set(string.punctuation) for char in token):
                continue

            # Lemmatize the token and yield
            lemma = lemmatize(token, tag)
            yield lemma


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i] 
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def topic_modeling(search_term, model="nmf"):
    # Download the pubmed data and vectorize it. A few heuristics are used
    # to filter out useless terms early on: the abstracts are stripped
    # common English words, words occurring in
    # only one document or in at least 95% of the documents are removed.
    print("Loading dataset...")
    file_name = search_term + "_texts.pickle"
    try:
        texts = pickle.load(open(file_name, "rb"))
    except IOError:
        texts = fetch_pubmed_abstracts(search_term)
        pickle.dump(texts, open(file_name, "wb"))

    n_samples = len(texts)

    if model == "nmf":
        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english',
                                           ngram_range=(1, 3),
                                           tokenizer=tokenize)
        tfidf = tfidf_vectorizer.fit_transform(texts)

        # Fit the NMF model
        print("Fitting the NMF model with tf-idf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features))
        nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
        nmf.fit(tfidf)

        print("\nTopics in NMF model:")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        print_top_words(nmf, tfidf_feature_names, n_top_words)
    elif model == "lda":
        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english',
                                        ngram_range=(1, 3),
                                        tokenizer=tokenize)
        tf = tf_vectorizer.fit_transform(texts)
        # Fit the LDA model
        print("Fitting LDA models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features))
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)


        print("\nTopics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words)
        # doc_topic_distrib = lda.transform(tf)
        # print(doc_topic_distrib)

def top_tfidf(search_term):
    print("Loading dataset...")
    file_name = search_term + "_texts.pickle"
    try:
        texts = pickle.load(open(file_name, "rb"))
    except IOError:
        texts = fetch_pubmed_abstracts(search_term)
        pickle.dump(texts, open(file_name, "wb"))

    n_samples = len(texts)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=10,
                                       max_features=n_features,
                                       stop_words='english',
                                       ngram_range=(1, 3),
                                       tokenizer=tokenize)

    tfidf = tfidf_vectorizer.fit_transform(texts)
    scores = zip(tfidf_vectorizer.get_feature_names(),
                 np.asarray(tfidf.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))

if __name__ == '__main__':
    search_term = SEARCH_TERMS[1]
    # top_tfidf(search_term)
    topic_modeling(search_term, model='nmf')
