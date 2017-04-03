import nltk
import re
import ssl
from nltk.stem.porter import *

def download_nltk_resources():
    """There is a known bug currently with nltk and ssl verification. This is a workaround via stackoverflow."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download()


def tokenize(sentence):
    ''' Takes in a sentence and returns the tokenized version (a list of words in the sentence). '''
    return nltk.word_tokenize(sentence)


def pos_tag_tokens(tokens):
    ''' Takes in a tokenized sentence and returns the pos tags for each token [word, token]. '''
    return nltk.pos_tag(tokens)


def noun_phrases(sentence):
    ''' Takes in a sentence and returns a list of all the noun phrases. '''
    noun_phrases = []
    tokens = tokenize(sentence)
    tokens_with_pos_tag = pos_tag_tokens(tokens)
    grammar = """
        NP: {<DT|PP\$>?<JJ>*<NN>}
            {<NNP>+}
            {<NN>+}
            {<PRP><NN>}
            {<PRP$>}
            {<NNP><NNS>}
            {<DT><NNP>}
            {<DT><CD><NNS>}
            {<CD><NNS>}
            {<NNS>+}
            {(<DT>?<RB>?)?<JJ|CD>*(<JJ|CD><,>)*<NN.*>+}
            {(<DT|PRP.>?<RB>?)?<JJ|CD>*(<JJ|CD><,>)*(<NN.*>)+}
            {<WP>}
            {<NNP><POS><NNP>}
        """
    chunker = nltk.RegexpParser(grammar)
    result = chunker.parse(tokens_with_pos_tag)

    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        noun_phrase = ""
        for word in subtree.leaves():
            noun_phrase += word[0] + " "
        noun_phrases.append(noun_phrase.strip())

    # Add all dates in the format dd/mm/yy as noun phrases
    match = re.findall(r'(\d+/\d+/\d+)', sentence)
    for m in match:
        noun_phrases.append(m)

    return noun_phrases


def split_to_sentences(document):
    ''' Takes in a document and returns a list of sentences. '''
    return nltk.tokenize.sent_tokenize(document)


def noun_phrases_in_document(document):
    ''' Takes in a document and returns a list of all the noun phrases. '''
    collected_noun_phrases = []
    
    for sentence in split_to_sentences(document):
       sentence_noun_phrases = noun_phrases(sentence)
       for np in sentence_noun_phrases:
           indexed_np = {'np': np, 'sentence': sentence}    
           collected_noun_phrases.append(indexed_np)
    return collected_noun_phrases
