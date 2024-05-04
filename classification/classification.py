import json, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def load_train_test_data(train_path, test_path):
    with open(train_path, 'r') as file:
        train_data = json.load(file)
    with open(test_path, 'r') as file:
        test_data = json.load(file)
    return train_data, test_data

def cut_suffix_from_word(word):
    suffixes = ['cu']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > 4:
            return word[:-len(suffix)] + "ti"
        
    suffixes = ['na', 'ni', 'no', 'ne']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
        
    # if word.startswith('ljubav'):
        # return 'ljubav'
    return word

def change_single_words_letter(word):
    # if word == 'gdje':
        # return 'gde'
    # if word == 'meni':
        # return 'mene'
    return word

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

def preprocess_x(text, stopwords):
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
    tokens = [cut_suffix_from_word(word) for word in tokens]
    tokens = [change_single_words_letter(word) for word in tokens]

    preprocessed_text = " ".join(tokens)
    return preprocessed_text

def preprocess_y(genre):
    return genre
    # if genre == 'pop':
        # return 0
    # if genre == 'rock':
        # return 1
    # return 2
    

def classify(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    classifier = SVC(kernel='rbf', C=1.5)
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
    

    
    