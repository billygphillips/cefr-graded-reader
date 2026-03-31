import pandas as pd
import spacy
from pathlib import Path

def load_ngsl(csv_path):
    """                                                                                                                                    
    Load the NGSL wordlist and return a dict of sets, one per frequency threshold.                                                         
    Each set contains all lemmas ranked <= that threshold.                                                                                 
    e.g. ngsl[800] = {'the', 'be', 'and', ...}  (top 800 most common words)                                                                
    """                                                                                                                                    
    df = pd.read_csv(csv_path)                                                                                                             
    ngsl = {}                                                                                                                              
    for threshold in [800, 1000, 2000, 3000]:                                                                                              
        ngsl[threshold] = set(df[df['SFI Rank'] <= threshold]['Lemma'].str.lower())                                                        
    return ngsl   

def load_spacy():                                                                                                                          
    """Load the spaCy English model."""                                                                                                    
    return spacy.load("en_core_web_sm") 

def extract_features(text, nlp, ngsl):
    """
    Extract CEFR-relevant features from a text string.
    Returns a dict of feature name → float value.
    """
    doc = nlp(text)

    # Get all tokens that are actual words (skip punctuation and whitespace)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    lemmas = [t.lemma_.lower() for t in tokens]

    if len(lemmas) == 0:
        return None

    # --- Feature 1: average sentence length ---
    sentences = list(doc.sents)
    avg_sent_len = len(tokens) / len(sentences)

    # --- Feature 2: NGSL vocabulary coverage ---
    ngsl_800  = sum(1 for l in lemmas if l in ngsl[800])  / len(lemmas)
    ngsl_1000 = sum(1 for l in lemmas if l in ngsl[1000]) / len(lemmas)
    ngsl_2000 = sum(1 for l in lemmas if l in ngsl[2000]) / len(lemmas)
    ngsl_3000 = sum(1 for l in lemmas if l in ngsl[3000]) / len(lemmas)

    # --- Feature 3: type-token ratio (lexical diversity) ---
    ttr = len(set(lemmas)) / len(lemmas)

    # --- Feature 4: syntactic complexity ---
    def tree_depth(token):
        depth = 0
        while token.head != token:
            token = token.head
            depth += 1
        return depth

    avg_tree_depth = sum(tree_depth(t) for t in doc) / len(list(doc))

    subord_deps = {'advcl', 'relcl', 'csubj', 'ccomp', 'xcomp', 'acl'}
    subord_ratio = sum(1 for t in doc if t.dep_ in subord_deps) / len(list(doc))

    # --- Feature 5: average word length (characters) ---
    avg_word_len = sum(len(t.text) for t in tokens) / len(tokens)

    # --- Feature 6: out-of-NGSL ratio ---
    # % of lemmas not found anywhere in the top-3000 NGSL
    out_of_ngsl = sum(1 for l in lemmas if l not in ngsl[3000]) / len(lemmas)

    return {
        'avg_sent_len':  avg_sent_len,
        'ngsl_800':      ngsl_800,
        'ngsl_1000':     ngsl_1000,
        'ngsl_2000':     ngsl_2000,
        'ngsl_3000':     ngsl_3000,
        'ttr':           ttr,
        'avg_tree_depth': avg_tree_depth,
        'subord_ratio':  subord_ratio,
        'avg_word_len':  avg_word_len,
        'out_of_ngsl':   out_of_ngsl,
    }