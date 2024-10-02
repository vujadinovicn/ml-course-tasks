import json, sys
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def load_train_test_data(train_path, test_path):
    with open(train_path, 'r') as file:
        train_data = json.load(file)
    with open(test_path, 'r') as file:
        test_data = json.load(file)
    return train_data, test_data
   
def get_all_stopwords():
    srb_stopwords = ['se', 'a', 'me', 'o', 'ako', 'ali',
                    'u', 'i', 'jos', 'al', 
                    'te', 'iz', 'uz', 'sto', 'oko',
                    'ili', 'gde', 'jer', 'k', 'l', 
                    'cu', 's', 'za', 'sam', 'dok', 
                    'iako', 'tako', 'na', 'ni',
                    'il', 'bi']
    return srb_stopwords

def preprocess_digits(word):
    for c in word:
        if c.isdigit():
            return ""
    return word

def preprocess_x(text, stopwords):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))

    tokens = text.split(" ")
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    tokens = [word.lower() for word in tokens if word not in stopwords]
    tokens = [preprocess_digits(word) for word in tokens if word != ""]

    preprocessed_text = " ".join(tokens)
    return preprocessed_text

def preprocess_y(genre):
    if genre == 'pop':
        return 0
    if genre == 'rock':
        return 1
    return 2

def classify(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    classifier = SVC(kernel='rbf', C=1.58, gamma=1.38)
    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)

    print(f1_score(y_test, y_pred, average='micro'))


if __name__ == "__main__":
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    train_data, test_data = load_train_test_data(train_dataset_path, test_dataset_path)

    stopwords = get_all_stopwords()

    X_train = [preprocess_x(entry['strofa'], stopwords) for entry in train_data]
    y_train = [preprocess_y(entry['zanr']) for entry in train_data]
    X_test = [preprocess_x(entry['strofa'], stopwords) for entry in test_data]
    y_test = [preprocess_y(entry['zanr']) for entry in test_data]

    classify(X_train, X_test, y_train, y_test)
    

    
    