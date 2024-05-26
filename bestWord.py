import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import pandas as pd

#reading data from csv
synonyms = pd.read_csv('synonyms_dictionary.csv')
target = pd.read_csv('target.csv')

#converting data in desired formate
output_word = target.columns.tolist()
synonyms_dict = {col: synonyms[col].dropna().tolist() for col in synonyms.columns}


#preprocessing function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

#output words
def outputWord(input_text):
    # Combine target words and their synonyms
    expanded_target_words = []
    for word in output_word:
        expanded_target_words.append(word)
        if word in synonyms_dict:
            expanded_target_words.extend(synonyms_dict[word])
    
    # Vectorize texts
    vectorizer = TfidfVectorizer().fit_transform([input_text] + expanded_target_words)
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
    
    # Find the best matching index
    best_index = cosine_similarities.argmax()
    
    # Map the index back to the target word
    best_match = expanded_target_words[best_index]
    
    for target_word in output_word:
        if best_match == target_word or best_match in synonyms_dict.get(target_word, []):
            return target_word
