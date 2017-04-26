import nlp_util
import io_util
import logistic_regression as lr


TRAIN_DATA_DIRECTORY = 'data/train/'
DEV_DATA_DIRECTORY = 'data/dev/'
DEV_PRED_DIRECTORY = 'data_pred/dev/'

vectorizer = None
tfidf_matrix = None

def main():
    global vectorizer
    global tfidf_matrix

    ''' Training Process for Task B '''
    # Read in the labeled phrases
    test_phrases = io_util.read_each_ann_file(TRAIN_DATA_DIRECTORY) 

    # Read in the documents and get a dictionary
    examples, documents = generate_example_dictionary(test_phrases)

    # Identify the sentence each labeled phrase came from
    set_sentence_field(examples)

    # Remove examples where the sentence couldn't be found
    remove_bad_examples(examples)

    # Create tfidf matrix from the corpora
    tfidf_matrix, vectorizer = nlp_util.get_tfidf_vectors(documents)

    # Create feature vectors for each training example
    training_vectors = create_feature_vectors(examples)

    # Write feature vectors to file
    io_util.write_feature_vectors('train_vectors.txt', training_vectors)

    # Train a logistic regression classifer
    model = lr.train() 

    ''' Testing Process for Task B '''
    # Read in the development phrases
    dev_phrases = io_util.read_each_ann_file(DEV_DATA_DIRECTORY)

    # Read in the documents and get a dictionary
    examples, documents = generate_example_dictionary(dev_phrases)

    # Identify the sentence each labeled phrase came from
    set_sentence_field(examples)

    # Remove examples where the sentence couldn't be found
    remove_bad_examples(examples)

    # Create tfidf matrix from the corpora
    tfidf_matrix, vectorizer = nlp_util.get_tfidf_vectors(documents)

    # Create feature vectors for each training example
    dev_vectors = create_feature_vectors(examples)

    # Make a prediction for each development example
    correct = 0
    total = len(dev_vectors)
    for vector in dev_vectors:
        prediction = model.predict([vector[1:]])
        if str(prediction[0]) == str(vector[0]):
            correct += 1

    print('Total:', total)
    print('Correct:', correct)


def create_feature_vectors(examples):
    global vectorizer
    global tfidf_matrix

    vectors = []
    document_number = 0
    for key, value in examples.items():
        for example in value['examples']:
            tfidf_score = nlp_util.get_tfidf_score(vectorizer, tfidf_matrix, example.key_phrase, document_number)
            vectors.append([example.label, tfidf_score])
        document_number += 1
    return vectors


def remove_bad_examples(examples):
    for key, value in examples.items():
        value['examples'][:] = [x for x in value['examples'] if x.key_phrase in x.sentence]


def generate_example_dictionary(labeled_phrases):
    ''' Returns a dictionary where they key is a document path and the value is 
        that document and labeled examples pertaining to that document. '''
    relevant_documents = set()
    documents = []
    document_dictionary = {}

    for phrase in labeled_phrases:
        relevant_documents.add(phrase.text_path)
    relevant_documents = list(relevant_documents)

    for document_path in relevant_documents:
        document = io_util.read_file(document_path)
        documents.append(document)
        document_dictionary[document_path] = {'document':document, 'examples':[]}

    for key in document_dictionary.keys():
        for labeled_phrase in labeled_phrases:
            if labeled_phrase.text_path == key:
                document_dictionary[key]['examples'].append(labeled_phrase)
    return document_dictionary, documents


def set_sentence_field(examples):
    ''' Sets the 'sentence' field of each labeled phrase. '''
    for key, value in examples.items():
        document_sentences = nlp_util.split_to_sentences(value['document'])
        sentences_indexed = []
        start_index = 0
        for sentence in document_sentences:
            sentences_indexed.append({'sentence':sentence, 'start_index':start_index})
            start_index += len(sentence)
        for example in value['examples']:
            example.sentence = sentences_indexed[-1]['sentence']
            prev_sentence = sentences_indexed[0]
            for index, sentence in enumerate(sentences_indexed):
                if example.start_index < sentence['start_index']:
                    example.sentence = prev_sentence['sentence']
                    break
                prev_sentence = sentence
                

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
            x = 5
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
