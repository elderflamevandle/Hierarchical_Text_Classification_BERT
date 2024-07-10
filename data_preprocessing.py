import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def preprocess_data(df):
    if 'Title' in df.columns and 'Text' in df.columns:
        df['text'] = df['Title'] + ' ' + df['Text']
        df['Text'] = df['Text'].apply(preprocess_text)
    elif 'text' not in df.columns:
        raise ValueError("DataFrame must contain either 'text' column or both 'Title' and 'Text' columns")

    required_columns = ['Cat1', 'Cat2', 'Cat3']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

    le1, le2, le3 = LabelEncoder(), LabelEncoder(), LabelEncoder()

    df['Cat1_encoded'] = le1.fit_transform(df['Cat1'])
    df['Cat2_encoded'] = le2.fit_transform(df['Cat2'])
    df['Cat3_encoded'] = le3.fit_transform(df['Cat3'])

    df['text'] = df['text'].fillna('')
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

    df = df[['text', 'Cat1_encoded', 'Cat2_encoded', 'Cat3_encoded']]

    return df, le1, le2, le3

def preprocess_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    df, le1, le2, le3 = preprocess_data(df)
    
    joblib.dump(le1, 'le1.joblib')
    joblib.dump(le2, 'le2.joblib')
    joblib.dump(le3, 'le3.joblib')
    
    return df, le1, le2, le3