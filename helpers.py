import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
  if not (text == "" or pd.isnull(text)): 
    # removing this weird URL tag I found in many of the examples
    text = re.sub(r'URL_[A-Za-z0-9]+', ' ', text)
    # removing non-alphanumeric characters
    return re.sub(r'[^A-Za-z0-9]+', ' ', text).lower().strip()

def get_model_input(text_input, vectorizer):
	text_input = clean_text(text_input)
	return vectorizer.transform([text_input])