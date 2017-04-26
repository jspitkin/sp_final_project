import sys
import glob
import os
import Keyphrase as kp

def read_each_ann_file(path):
    labeled_examples = []
    for filename in glob.glob(os.path.join(path, '*.ann')):
        phrases = parse_phrases_ann_file(filename)
        labeled_examples.extend(phrases)
    return labeled_examples


def parse_phrases_ann_file(path):
    key_phrases = []
    with open(path, 'r') as ann_file:
        text_path = path.strip('.ann') + '.txt'
        for line in ann_file:
            phrase = kp.Keyphrase()
            line = line.split('\t')
            if line[0][0] != 'T':
                continue
            phrase.ID = line[0]
            phrase.key_phrase = line[2].strip('\n')
            line = line[1].split()
            phrase.label = line[0]
            if phrase.label == 'Material':
                phrase.label = 0
            elif phrase.label == 'Process':
                phrase.label = 1
            elif phrase.label == 'Task':
                phrase.label = 2
            phrase.start_index = int(line[1])
            if len(line) == 4:
                phrase.end_index = int(line[3])
            else:
                phrase.end_index = int(line[2])
            phrase.text_path = text_path
            key_phrases.append(phrase)
    return key_phrases

def read_file(path):
    with open(path, 'r') as f:
        return f.read().replace('\n', '')

def write_feature_vectors(path, vectors):
    with open(path, 'w') as feature_vector_file:
        for vector in vectors:
            for entry in vector:
                feature_vector_file.write(str(entry))
                feature_vector_file.write(' ')
            feature_vector_file.write('\n')
