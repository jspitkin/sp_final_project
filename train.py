import nlp_util
import io_util


TRAIN_DATA_DIRECTORY = 'data/train/'
DEV_DATA_DIRECTORY = 'data/dev/'
DEV_PRED_DIRECTORY = 'data_pred/dev/'


def main():
    ''' Program entry point. '''
    # Read in the labeled phrases
    labeled_phrases = io_util.read_each_ann_file(TRAIN_DATA_DIRECTORY) 

    # Read in the documents and get a dictionary
    documents = generate_document_dictionary(labeled_phrases)

    # Identify the sentence each labeled phrase came from
    set_sentence_field(labeled_phrases, documents)

    # Extract POS features for each labeled phrase
    possible_features = generate_possible_features(labeled_phrases)
    generate_feature_vectors(labeled_phrases, possible_features)


def generate_document_dictionary(labeled_phrases):
    ''' Returns a dictionary where they key is a document path and the value is 
        that document. '''
    relevant_documents = set()
    document_dictionary = {}

    for phrase in labeled_phrases:
        relevant_documents.add(phrase.text_path)
    relevant_documents = list(relevant_documents)

    for document_path in relevant_documents:
        document = io_util.read_file(labeled_phrases[0].text_path)
        document_dictionary[document_path] = document
    return document_dictionary


def set_sentence_field(phrases, documents):
    ''' Sets the 'sentence' field of each labeled phrase. '''
    for phrase in phrases:
        document_sentences = nlp_util.split_to_sentences(documents[phrase.text_path])
        print(document_sentences)

def generate_possible_features(phrases):
    features = {}
    unique_head_nouns = get_unique_head_nouns(phrases)
    unique_pos_tags = get_unique_pos_tags(phrases)
    return features

def get_unique_pos_tags(phrases):
    ''' Takes in a list of phrases and returns a list of unique pos tags. '''
    unique_pos_tags = set()
    for phrase in phrases:
        if phrase.sentence is None:
            print(phrase, ' bad')
        #sentence_pos_tags = nlp_util.pos_tag_tokens(nlp_util.tokenize(phrase.sentence))
        #print(sentence_pos_tags)


def get_unique_head_nouns(phrases):
    ''' Takes in a list of phrases and returns a list of unique head nouns.
        Using the simple heuristic of assuming the left-most word is the head noun. '''
    unique_head_nouns = set()
    for phrase in phrases:
        unique_head_nouns.add(phrase.key_phrase.split()[-1])
    return list(unique_head_nouns)


def generate_feature_vectors(phrases, possible_features):
    feature_vectors = []
    for phrase in phrases:
        feature_vector = {'label': phrase.label, 'features': []}



def in_bounds(inner_start, inner_end, outer_start, outer_end):
    ''' Returns true if the inner bounds lie within the outer bounds.
        Allows for a degree of error eta. '''
    eta = 5
    return inner_start + eta >= outer_start and inner_end - eta <= outer_end


if __name__ == '__main__':
    main()
