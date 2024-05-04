import json
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.pipeline import make_pipeline
import numpy as np
# from nltk.stem.porter import PorterStemmer

def get_encoding_by_type(encoding_type):
    if encoding_type == 'TFIDF':
        vectorizer = TfidfVectorizer()
    elif encoding_type == 'BOW':
        vectorizer = CountVectorizer()
    return vectorizer

def get_classifier_by_type(classifier_type):
    if classifier_type == 'SVM':
        classifier = SVC(kernel='rbf', C=1.5)
    elif classifier_type == 'NB':
        classifier = MultinomialNB()
    elif classifier_type == 'LogReg':
        classifier = LogisticRegression(max_iter=1000)
    elif classifier_type == 'Perceptron':
        classifier = Perceptron(max_iter=1000)
    return classifier

def classify(X_train, X_test, y_train, y_test, encoding_type, classifier_type):
    vectorizer = get_encoding_by_type(encoding_type)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    classifier = get_classifier_by_type(classifier_type)
    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    print(f"F1-score for {classifier_type} with {encoding_type}: {f1_score(y_test, y_pred, average='micro')}")


def all_tfidf_classify(X_train, X_test, y_train, y_test):
    classify(X_train, X_test, y_train, y_test, 'TFIDF', 'SVM')
    # classify(X_train, X_test, y_train, y_test, 'TFIDF', 'NB')
    # classify(X_train, X_test, y_train, y_test, 'TFIDF', 'LogReg')
    # classify(X_train, X_test, y_train, y_test, 'TFIDF', 'Perceptron')

def all_bow_classify(X_train, X_test, y_train, y_test):
    classify(X_train, X_test, y_train, y_test, 'BOW', 'SVM')
    classify(X_train, X_test, y_train, y_test, 'BOW', 'NB')
    classify(X_train, X_test, y_train, y_test, 'BOW', 'LogReg')
    classify(X_train, X_test, y_train, y_test, 'BOW', 'Perceptron')

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def cross_validation(X, y, encoding_type, classifier_type):
    vectorizer = get_encoding_by_type(encoding_type)
    classifier = get_classifier_by_type(classifier_type)
    
    X = np.array(X)
    y = np.array(y)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)
    
    f1_micro_scores = []
    
    for train_index, test_index in cv.split(X, y):
        train_index = train_index.astype(int)
        test_index = test_index.astype(int)  
        
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        classifier.fit(X_train_bow, y_train)
        y_pred = classifier.predict(X_test_bow)
        
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_micro_scores.append(f1_micro)
    
    print(f"Mean F1-score for {classifier_type} with {encoding_type}: {np.mean(f1_micro_scores)}")
    return f1_micro_scores


def cross_validation_old_incorrect(X, y, encoding_type, classifier_type):
    vectorizer = get_encoding_by_type(encoding_type)
    X_bow = vectorizer.fit_transform(X)
    classifier = get_classifier_by_type(classifier_type)
    pipeline = make_pipeline(vectorizer, classifier)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_micro')
    # print(f"CV F1 scores for {classifier_type} with {encoding_type}:", cv_scores)
    print(f"Mean F1-score for {classifier_type} with {encoding_type}: {np.mean(cv_scores)}")

def all_tfidf_cross_validation(X, y):
    cross_validation(X, y, 'TFIDF', 'SVM')
    # cross_validation(X, y, 'TFIDF', 'NB')
    # cross_validation(X, y, 'TFIDF', 'LogReg')
    # cross_validation(X, y, 'TFIDF', 'Perceptron')

def all_bow_cross_validation(X, y):
    cross_validation(X, y, 'BOW', 'SVM')
    cross_validation(X, y, 'BOW', 'NB')
    cross_validation(X, y, 'BOW', 'LogReg')
    cross_validation(X, y, 'BOW', 'Perceptron')

def serbian_lemmatize(word):
    suffixes = ['cu']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > 4:
            # pass
            # print(word)
            return word[:-len(suffix)] + "ti"
        
    suffixes = ['na', 'ni', 'no', 'ne']
    for suffix in suffixes:
        if word.endswith(suffix):
            # pass
            # print(word)
            return word[:-len(suffix)]
    # if word.startswith('ljubav'):
        # return 'ljubav'
    return word

def ijekavica(word):
    # if word == 'gdje':
        # return 'gde'
    # if word == 'meni':
        # return 'mene'
    return word

def preprocess(text, stopwords):
    import re, string

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    # text = re.sub(r'\d+(\.\d+)?', "", text)

    tokens = text.split(" ")
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    tokens = [word.lower() for word in tokens if word not in stopwords]
    tokens = [serbian_lemmatize(word) for word in tokens]
    tokens = [ijekavica(word) for word in tokens]

    preprocessed_text = " ".join(tokens)

    return preprocessed_text

def log_occurences():
    folk_words = {}
    all_folk_texts = []
    for i in range(len(data)):
        if data[i]['zanr'] == 'pop':
            all_folk_texts.append(preprocess(data[i]['strofa'], stopwords))
    for text in all_folk_texts:
        words = text.split(" ")
        for word in words:
            try:
                folk_words[word] += 1
            except:
                folk_words[word] = 1
    sorted_dict = dict(sorted(folk_words.items(), key=lambda item: item[1], reverse=True))
    with open("classification/data/occurs_pop.json", 'w') as json_file:
        json.dump(sorted_dict, json_file, indent=2)

def preprocess_genre(genre):
    if genre == 'pop':
        return 0
    if genre == 'rock':
        return 1
    if genre == 'folk':
        return 2
    
def get_all_stopwords():
    srb_stopwords = ['se', 'a', 'me', 'o', 'ako', 'ali',
                    'u', 'i', 'takodje', 'jos',
                    'te', 'iz', 'uz', 'sto', 'oko',
                    'ili', 'gde', "svi", 'jer', 'k', 'l', 
                    'niti', 'treba', 'trebalo', 'trebala', 
                    'trebaju', 'trebas', 'trebam', 'trebao',
                    'cu', 's', 'za', 'sam', 'dok']
    
    eng_stopwords = ['me', 'my', 'myself', 'we', 'our', 'ours',
                    'ourselves', 'you', 'your', 'yours', 'yourself',
                    'yourselves', 'he', 'him', 'his', 'himself', 
                    'she', 'her', 'hers', 'herself', 'it', 'its',
                    'itself', 'they', 'them', 'their', 'theirs',
                    'themselves', 'what', 'which', 'who', 'whom',
                    'this', 'that', 'these', 'those', 'am', 'is',
                    'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'having', 'do', 'does', 
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but',
                    'if', 'or', 'because', 'as', 'until', 'while',
                    'of', 'at', 'by', 'for', 'with', 'about', 
                    'against', 'between', 'into', 'through', 'during',
                    'before', 'after', 'above', 'below', 'to', 'from',
                    'up', 'down', 'in', 'out', 'on', 'off', 'over',
                    'under', 'again', 'further', 'then', 'once', 'here',
                    'there', 'when', 'where', 'why', 'how', 'all', 'any',
                    'both', 'each', 'few', 'more', 'most', 'other', 
                    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                    'same', 'so', 'than', 'too', 'very', 's', 't', 
                    'can', 'will', 'just', 'don', 'should', 'now']
    
    stopwords = srb_stopwords + eng_stopwords
    return stopwords
    
if __name__ == "__main__":
    train_path = 'classification/data/train.json'
    with open(train_path, 'r') as file:
        data = json.load(file)
    
    stopwords = get_all_stopwords()

    # ucitavamo sve koje nismo iskoristili do sad
    up_st = []
    with open('classification/data/serbian_stopwords.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.rstrip() not in stopwords:
                up_st.append(line.rstrip())

    for i in range(len(up_st)):
        stopwordss = stopwords + [up_st[i]]

        X = [preprocess(entry['strofa'], stopwordss) for entry in data]
        y = [preprocess_genre(entry['zanr']) for entry in data]

        print(f'Kada dodamo rec: ' + up_st[i] + '...')
        all_tfidf_cross_validation(X, y)