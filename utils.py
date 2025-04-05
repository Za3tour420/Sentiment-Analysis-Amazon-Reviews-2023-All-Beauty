import re
from string import punctuation
import nltk
from nltk.corpus import stopwords
import joblib

nltk.download('stopwords')

def remove_emojis(data: str):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                    "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def remove_stopwords(data: str):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    stop_words
    stop_words.discard('not') # 'not', 'but' and 'such' are important to keep
    stop_words.discard('such')
    stop_words.discard('but')
    return ' '.join([word for word in data.split() if word not in stop_words])

def remove_punctuation(data: str):
    clean_text = data.translate(str.maketrans('', '', punctuation))
    return clean_text

def load_model(file_path: str):
    try:
        print(f"Trying to load model from file {file_path}...")
        model = joblib.load(file_path)
        print("Loaded model successfully!")
        return model
    except FileNotFoundError:
        print("Failed to load model from file. Create the model and train it!")
        return None