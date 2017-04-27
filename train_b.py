import nlp_util
import io_util
import logistic_regression as lr


TRAIN_DATA_DIRECTORY = 'data/train/'
DEV_DATA_DIRECTORY = 'data/dev/'

MATERIAL = 0
PROCESS = 1
TASK = 2

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
    io_util.write_feature_vectors('train_vectors_b.txt', training_vectors)

    # Train a logistic regression classifer
    model = lr.train('train_vectors_b.txt') 

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
    evaluate(dev_vectors, model)


def create_feature_vectors(examples):
    global vectorizer
    global tfidf_matrix

    vectors = []
    document_number = 0
    for key, value in examples.items():
        for example in value['examples']:
            vector = {'label':example.label}
            # tfidf score features
            tfidf_score = nlp_util.get_tfidf_score(vectorizer, tfidf_matrix, example.key_phrase, document_number)
            vector['tfidf_score'] = tfidf_score
            vectors.append(vector)
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
    unique_head_nouns = get_unique_head_nouns(phrases)
    unique_pos_tags = get_unique_pos_tags(phrases)
    return features


def get_unique_pos_tags(phrases):
    ''' Takes in a list of phrases and returns a list of unique pos tags. '''
    unique_pos_tags = set()
    for phrase in phrases:
        sentence_pos_tags = nlp_util.pos_tag_tokens(nlp_util.tokenize(phrase.sentence))
        for pos_tag in sentence_pos_tags:
            unique_pos_tags.add(pos_tag[1])
    print(len(unique_pos_tags))


def generate_feature_vectors(phrases, possible_features):
    feature_vectors = []
    for phrase in phrases:
        feature_vector = {'label': phrase.label, 'features': []}


def evaluate(dev_vectors, model):
    process_correct = 0
    process_predictions = 0
    process_count = 0

    task_correct = 0
    task_predictions = 0
    task_count = 0

    material_correct = 0
    material_predictions = 0
    material_count = 0

    total = len(dev_vectors)
    total_correct = 0

    for vector in dev_vectors:
        prediction = model.predict([vector[1:]])
        # Calculate correct predictions
        if int(prediction[0]) == MATERIAL and vector[0] == MATERIAL:
            material_correct += 1
            total_correct += 1
        elif int(prediction[0]) == PROCESS and vector[0] == PROCESS:
            process_correct += 1
            total_correct += 1
        elif int(prediction[0]) == TASK and vector[0] == TASK:
            task_correct += 1
            total_correct += 1
        # Calculate ground truth counts
        if int(vector[0]) == MATERIAL:
            material_count += 1
        elif int(vector[0]) == PROCESS:
            process_count += 1
        elif int(vector[0]) == TASK:
            task_count += 1
        # Calculate prediction counts
        if int(prediction[0]) == MATERIAL:
            material_predictions += 1
        elif int(prediction[0]) == PROCESS:
            process_predictions += 1
        elif int(prediction[0]) == TASK:
            task_predictions += 1

    #print(task_correct, task_predictions, task_count)
    #print(material_correct, material_predictions, material_count)
    #print(process_correct, process_predictions, process_count)
    
    if process_count == 0:
        process_recall = 0
    else:
        process_recall = process_correct / process_count
    if process_predictions:
        process_precision = 0
    else:
        process_precision = process_correct / process_predictions
    if process_recall == 0 and process_precision == 0:
        process_f1 = 0
    else:
        process_f1 = (2 * process_recall * process_precision) / (process_recall + process_precision)

    if task_count == 0:
        task_recall = 0
    else:
        task_recall = task_correct / task_count
    if task_predictions == 0:
        task_precision = 0
    else:
        task_precision = task_correct / task_predictions
    if task_recall == 0 and task_precision == 0:
        task_f1 = 0
    else:
        task_f1 = (2 * task_recall * task_precision) / (task_recall + task_precision)

    if material_count == 0:
        material_recall = 0
    else:
        material_recall = material_correct / material_count
    if material_predictions == 0:
        material_precision = 0
    else:
        material_precision = material_correct / material_predictions
    if material_precision == 0 and material_recall == 0:
        material_f1 = 0
    else:
        material_f1 = (2 * material_recall * material_precision) / (material_recall + material_precision)

    overall_accuracy = total_correct / total

    print()
    print('Process')
    print('Precision:', process_precision, '\tRecall:', process_recall, '\tF1-Score:', process_f1)
    print()
    print('Task')
    print('Precision:', task_precision, '\tRecall:', task_recall, '\tF1-Score:', task_f1)
    print()
    print('Material')
    print('Precision:', material_precision, '\tRecall:', material_recall, '\tF1-Score:', material_f1)
    print()
    print('Overall Accuracy:', overall_accuracy)


if __name__ == '__main__':
    main()
