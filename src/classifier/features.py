import pandas as pd
import spacy

# Connector tiers aligned to CEFR grammar syllabus
CONNECTORS = {
    # Tier 1 — A2: basic coordinators and simple subordinators
    'and': 1, 'but': 1, 'or': 1, 'so': 1, 'because': 1,
    # Tier 2 — B1: common complex connectors
    'although': 2, 'however': 2, 'while': 2, 'unless': 2, 'therefore': 2,
    # Tier 3 — B2+: academic / formal connectors
    'furthermore': 3, 'nevertheless': 3, 'whereas': 3, 'moreover': 3, 'consequently': 3,
}


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

    Feature set (14 features):
      Lexical:   avg_sent_len, ngsl_800/1000/2000/3000, avg_word_len, out_of_ngsl
      Syntactic: subord_ratio, relative_clause_rate
      Grammar:   passive_rate, perfect_rate, modal_diversity,
                 conditional_rate, connector_tier
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    n_sents = max(len(sentences), 1)

    # Content tokens only (skip punctuation and whitespace)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    lemmas = [t.lemma_.lower() for t in tokens]
    n_tokens = len(lemmas)

    if n_tokens == 0:
        return None

    # ── Lexical features ────────────────────────────────────────────────────

    avg_sent_len = n_tokens / n_sents

    ngsl_800  = sum(1 for l in lemmas if l in ngsl[800])  / n_tokens
    ngsl_1000 = sum(1 for l in lemmas if l in ngsl[1000]) / n_tokens
    ngsl_2000 = sum(1 for l in lemmas if l in ngsl[2000]) / n_tokens
    ngsl_3000 = sum(1 for l in lemmas if l in ngsl[3000]) / n_tokens

    avg_word_len = sum(len(t.text) for t in tokens) / n_tokens

    out_of_ngsl = sum(1 for l in lemmas if l not in ngsl[3000]) / n_tokens

    # ── Syntactic features ───────────────────────────────────────────────────

    # Subordinate clause ratio (advcl, ccomp, xcomp, acl, csubj — NOT relcl, pulled out separately)
    subord_deps = {'advcl', 'csubj', 'ccomp', 'xcomp', 'acl'}
    n_doc_tokens = len(list(doc))
    subord_ratio = sum(1 for t in doc if t.dep_ in subord_deps) / n_doc_tokens

    # Relative clause rate — B1+ marker, per sentence
    relcl_count = sum(1 for t in doc if t.dep_ == 'relcl')
    relative_clause_rate = relcl_count / n_sents

    # ── Grammar features ────────────────────────────────────────────────────

    # Passive voice — nsubjpass/auxpass dependency labels
    passive_count = sum(1 for t in doc if t.dep_ in {'nsubjpass', 'auxpass'})
    passive_rate = passive_count / n_sents

    # Perfect aspect — "have/has/had" as aux + head is past participle (VBN)
    perfect_count = sum(
        1 for t in doc
        if t.dep_ == 'aux' and t.lemma_.lower() == 'have' and t.head.tag_ == 'VBN'
    )
    perfect_rate = perfect_count / n_sents

    # Modal diversity — count of DISTINCT modal lemmas (MD tag)
    # A2 allows: can, must, will (max 3). B1 adds: should, could. B2+: might, would, shall.
    modal_lemmas = [t.lemma_.lower() for t in doc if t.tag_ == 'MD']
    modal_diversity = float(len(set(modal_lemmas)))

    # Conditional rate — "if" as a subordinating marker (dep_ == "mark"), per sentence
    conditional_count = sum(
        1 for t in doc
        if t.lemma_.lower() == 'if' and t.dep_ == 'mark'
    )
    conditional_rate = conditional_count / n_sents

    # Connector tier — average CEFR tier of connectors found in the text
    connector_tiers = [CONNECTORS[t.lemma_.lower()] for t in doc if t.lemma_.lower() in CONNECTORS]
    connector_tier = sum(connector_tiers) / len(connector_tiers) if connector_tiers else 0.0

    return {
        'avg_sent_len':        avg_sent_len,
        'ngsl_800':            ngsl_800,
        'ngsl_1000':           ngsl_1000,
        'ngsl_2000':           ngsl_2000,
        'ngsl_3000':           ngsl_3000,
        'avg_word_len':        avg_word_len,
        'out_of_ngsl':         out_of_ngsl,
        'subord_ratio':        subord_ratio,
        'relative_clause_rate': relative_clause_rate,
        'passive_rate':        passive_rate,
        'perfect_rate':        perfect_rate,
        'modal_diversity':     modal_diversity,
        'conditional_rate':    conditional_rate,
        'connector_tier':      connector_tier,
    }
