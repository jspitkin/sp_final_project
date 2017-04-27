import nlp_util
import io_util
import logistic_regression as lr
import word_2_vec as w2v
import pdb


TRAIN_DATA_DIRECTORY = 'data/train/'
DEV_DATA_DIRECTORY = 'data/dev/'
TEST_DATA_DIRECTORY = 'data/test/'

SYNONYM = 0
HYPONYM = 1
NEITHER = 2

vectorizer = None
tfidf_matrix = None

def main():
    global vectorizer
    global tfidf_matrix

    ''' Training Process for Task C '''
    # Read in the labeled phrases
    test_phrases = io_util.read_each_ann_file(TRAIN_DATA_DIRECTORY) 

    # Read in the documents and get a dictionary
    examples, documents = generate_example_dictionary(test_phrases)

    # Identify the sentence each labeled phrase came from
    set_sentence_field(examples)

    # Remove examples where the sentence couldn't be found
    remove_bad_examples(examples)

    # Read in the labeled relations
    relations = io_util.read_each_ann_file_rel(TRAIN_DATA_DIRECTORY, examples)

    # Read in the documents and create a relations dictionary
    rel_examples, documents = generate_example_dictionary(relations)

    # Remove any examples that don't have the key phrases set
    for key, value in rel_examples.items():
        value['examples'][:] = [x for x in value['examples'] if x.key_phrase1 is not None and x.key_phrase2 is not None]

    # Create tfidf matrix from the corpora
    tfidf_matrix, vectorizer = nlp_util.get_tfidf_vectors(documents)

    # Create features for each training example
    training_features = create_features(rel_examples)
    add_casing_features(training_features)
    add_sim_features(training_features)

    # Create feature vector from features
    training_vectors = create_feature_vectors(training_features)

    # Write feature vectors to file
    io_util.write_feature_vectors('train_vectors_c.txt', training_vectors)

    # Train a logistic regression classifer
    model = lr.train('train_vectors_c.txt') 

    ''' Testing Process for Task C '''
    # Read in the development phrases
    dev_phrases = io_util.read_each_ann_file(DEV_DATA_DIRECTORY)

    # Read in the documents and get a dictionary
    examples, documents = generate_example_dictionary(dev_phrases)

    # Identify the sentence each labeled phrase came from
    set_sentence_field(examples)

    # Remove examples where the sentence couldn't be found
    remove_bad_examples(examples)

    # Read in the labeled relations
    relations = io_util.read_each_ann_file_rel(DEV_DATA_DIRECTORY, examples)

    # Read in the documents and create a relations dictionary
    rel_examples, documents = generate_example_dictionary(relations)

    # Remove any examples that don't have the key phrases set
    for key, value in rel_examples.items():
        value['examples'][:] = [x for x in value['examples'] if x.key_phrase1 is not None and x.key_phrase2 is not None]

    # Create tfidf matrix from the corpora
    tfidf_matrix, vectorizer = nlp_util.get_tfidf_vectors(documents)

    # Create features for each training example
    dev_features = create_features(rel_examples)
    add_casing_features(dev_features)
    add_sim_features(dev_features)

    # Create feature vector from features
    dev_vectors = create_feature_vectors(dev_features)

    # Make a prediction for each development example
    evaluate(dev_vectors, model)


def create_feature_vectors(examples):
    feature_vectors = [[example['label']] for example in examples]
    ignore_list = set(['sentence_1', 'sentence_2', 'arg_1', 'arg_2', 'key_phrase_1', 'key_phrase_2', 'label'])
    for index, example in enumerate(examples):
        sorted(example)
        for key, value in example.items():
            if key not in ignore_list:
                feature_vectors[index].append(value)
    return feature_vectors

def create_features(examples):
    global vectorizer
    global tfidf_matrix

    vectors = []
    document_number = 0
    for key, value in examples.items():
        for example in value['examples']:
            vector = {'label':example.label}
            vector['sentence_1'] = example.sentence1
            vector['sentence_2'] = example.sentence2
            vector['arg_1'] = example.Arg1
            vector['arg_2'] = example.Arg2
            vector['key_phrase_1'] = example.key_phrase1
            vector['key_phrase_2'] = example.key_phrase2
            # tfidf score features
            tfidf_score_1 = nlp_util.get_tfidf_score(vectorizer, tfidf_matrix, example.key_phrase1, document_number)
            tfidf_score_2 = nlp_util.get_tfidf_score(vectorizer, tfidf_matrix, example.key_phrase2, document_number)
            vector['tfidf_score_1'] = tfidf_score_1
            vector['tfidf_score_2'] = tfidf_score_2
            vector['tfidf_combined'] = tfidf_score_1 + tfidf_score_2
            vectors.append(vector)
        document_number += 1
    return vectors


def add_casing_features(vectors):
    for vector in vectors:
        vector['all_caps_1'] = 0
        vector['all_caps_2'] = 0
        vector['both_all_caps'] = 0
        vector['contains_cap_1'] = 0
        vector['contains_cap_2'] = 0
        vector['both_contains_cap'] = 0
        vector['contains_digit_1'] = 0
        vector['contains_digit_2'] = 0
        vector['both_digit'] = 0
        if any(char.isdigit() for char in vector['key_phrase_1']):
            vector['contains_digit_1'] = 1
        if any(char.isdigit() for char in vector['key_phrase_2']):
            vector['contains_digit_2'] = 1
        if vector['contains_digit_1'] + vector['contains_digit_2'] == 2:
            vector['both_digit'] = 1
        for word in vector['key_phrase_1'].split():
            if word[0].isupper():
                vector['contains_cap_1'] = 1
        for word in vector['key_phrase_2'].split():
            if word[0].isupper():
                vector['contains_cap_2'] = 1
        if vector['contains_cap_1'] + vector['contains_cap_2'] == 2:
            vector['both_contains_cap'] = 1
        all_caps_1 = True
        for letter in vector['key_phrase_1']:
            if letter.isalpha() and letter.islower():
                all_caps_1 = False
        if all_caps_1:
            vector['all_caps_1'] = 1
        all_caps_2 = True
        for letter in vector['key_phrase_2']:
            if letter.isalpha() and letter.islower():
                all_caps_2 = False
        if all_caps_2:
            vector['all_caps_2'] = 1
        if vector['all_caps_1'] + vector['all_caps_2'] == 2:
            vector['both_all_caps'] = 1


def add_sim_features(vectors):
    model = w2v.get_model()
    for vector in vectors:
        vector['len_phrase_1'] = 0
        for word in vector['key_phrase_1'].split():
            vector['len_phrase_1'] += len(word)
        vector['len_phrase_2'] = 0
        for word in vector['key_phrase_1'].split():
            vector['len_phrase_2'] += len(word)
        vector['phrases_same_sent'] = 0
        if vector['sentence_1'] == vector['sentence_2']:
            vector['phrases_same_sent'] = 1
        vector['cos_sim'] = w2v.get_cosine_sim(vector['key_phrase_1'], vector['key_phrase_2'], model)


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


def evaluate(dev_vectors, model):
    hyponym_correct = 0
    hyponym_predictions = 0
    hyponym_count = 0

    synonym_correct = 0
    synonym_predictions = 0
    synonym_count = 0

    neither_correct = 0
    neither_predictions = 0
    neither_count = 0

    total = len(dev_vectors)
    total_correct = 0

    for vector in dev_vectors:
        prediction = model.predict([vector[1:]])
        # Calculate correct predictions
        if int(prediction[0]) == HYPONYM and vector[0] == HYPONYM:
            hyponym_correct += 1
            total_correct += 1
        elif int(prediction[0]) == SYNONYM and vector[0] == SYNONYM:
            synonym_correct += 1
            total_correct += 1
        elif int(prediction[0]) == NEITHER and vector[0] == NEITHER:
            neither_correct += 1
            total_correct += 1
        # Calculate ground truth counts
        if int(vector[0]) == HYPONYM:
            hyponym_count += 1
        elif int(vector[0]) == SYNONYM:
            synonym_count += 1
        elif int(vector[0]) == NEITHER:
            neither_count += 1
        # Calculate prediction counts
        if int(prediction[0]) == HYPONYM:
            hyponym_predictions += 1
        elif int(prediction[0]) == SYNONYM:
            synonym_predictions += 1
        elif int(prediction[0]) == NEITHER:
            neither_predictions += 1
    
    if hyponym_count == 0:
        hyponym_recall = 0
    else:
        hyponym_recall = hyponym_correct / hyponym_count
    if hyponym_predictions == 0:
        hyponym_precision = 0
    else:
        hyponym_precision = hyponym_correct / hyponym_predictions
    if hyponym_recall == 0 and hyponym_precision == 0:
        hyponym_f1 = 0
    else:
        hyponym_f1 = (2 * hyponym_recall * hyponym_precision) / (hyponym_recall + hyponym_precision)

    if synonym_count == 0:
        synonym_recall = 0
    else:
        synonym_recall = synonym_correct / synonym_count
    if synonym_predictions == 0:
        synonym_precision = 0
    else:
        synonym_precision = synonym_correct / synonym_predictions
    if synonym_recall == 0 and synonym_precision == 0:
        synonym_f1 = 0
    else:
        synonym_f1 = (2 * synonym_recall * synonym_precision) / (synonym_recall + synonym_precision)

    if neither_count == 0:
        neither_recall = 0
    else:
        neither_recall = neither_correct / neither_count
    if neither_predictions == 0:
        neither_precision = 0
    else:
        neither_precision = neither_correct / neither_predictions
    if neither_precision == 0 and neither_recall == 0:
        neither_f1 = 0
    else:
        neither_f1 = (2 * neither_recall * neither_precision) / (neither_recall + neither_precision)

    overall_accuracy = total_correct / total

    print()
    print('Hyponym-of - ', hyponym_count, 'examples')
    print('Predictions:', hyponym_predictions, ' Precision:', hyponym_precision, ' Recall:', hyponym_recall, ' F1-Score:', hyponym_f1)
    print()
    print('Synonym-of - ', synonym_count, 'examples')
    print('Predictions:', synonym_predictions, ' Precision:', synonym_precision, ' Recall:', synonym_recall, ' F1-Score:', synonym_f1)
    print()
    print('Neither - ', neither_count, 'examples')
    print('Predictions:', neither_predictions, ' Precision:', neither_precision, ' Recall:', neither_recall, ' F1-Score:', neither_f1)
    print()
    print('Overall Accuracy:', overall_accuracy)


if __name__ == '__main__':
    main()
