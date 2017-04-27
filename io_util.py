import sys
import glob
import os
import Keyphrase as kp
import Relation as rl
from random import *

def read_each_ann_file(path):
    labeled_examples = []
    for filename in glob.glob(os.path.join(path, '*.ann')):
        phrases = parse_phrases_ann_file(filename)
        labeled_examples.extend(phrases)
    return labeled_examples


def read_each_ann_file_rel(path, rdict):
    labeled_examples = []
    for filename in glob.glob(os.path.join(path, '*.ann')):
        relations = parse_relations_ann_file(filename, rdict)
        labeled_examples.extend(relations)
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

def parse_relations_ann_file(path, rdict):
    relations = []
    Negid = 1
    with open(path, 'r') as ann_file:
        text_path = path.strip('.ann') + '.txt'
        examples = rdict[text_path]['examples']
        for line in ann_file:
            relationship = rl.Relation()
            line = line.split('\t')
            if line[0][0] == 'T':
                continue
            if line[0][0] == 'R':
                relationship.ID = line[0]
                relationship.label = 1 # Hyponym
                line = line[1].split()
                A1 = line[1].split(':')
                relationship.Arg1 = A1[1]
                A2 = line[2].split(':')
                relationship.Arg2 = A2[1]
                relationship.text_path = text_path
                NegRel = rl.Relation()
            if line[0][0] == '*':
                relationship.ID = line[0]
                relationship.label = 0 # Synonym
                line = line[1].split()
                relationship.Arg1 = line[1]
                relationship.Arg2 = line[2]
                relationship.text_path = text_path
                NegRel = rl.Relation()
            negPhr = []
            for phrase in examples:
                if relationship.Arg1 == phrase.ID:
                    relationship.sentence1 = phrase.sentence
                    relationship.key_phrase1 = phrase.key_phrase
                elif relationship.Arg2 == phrase.ID:
                    relationship.sentence2 = phrase.sentence
                    relationship.key_phrase2 = phrase.key_phrase
                else:
                    negPhr.append(phrase)
            count = 0
            i = 1
            p1 = randint(0, int(len(negPhr)/4))
            while count == 0:
                p2 = p1 + i
                if p2 >= len(negPhr):
                    break
                if negPhr[p1].label == negPhr[p2].label:
                    i += 1
                else:
                    NegRel.ID = "N" + str(Negid) + ""
                    Negid += 1
                    NegRel.label = 2
                    NegRel.key_phrase1 = negPhr[p1].key_phrase
                    NegRel.sentence1 = negPhr[p1].sentence
                    NegRel.Arg1 = negPhr[p1].ID
                    NegRel.key_phrase2 = negPhr[p2].key_phrase
                    NegRel.sentence2 = negPhr[p2].sentence
                    NegRel.Arg2 = negPhr[p2].ID
                    NegRel.text_path = negPhr[p2].text_path
                    relations.append(NegRel)
                    count += 1
            relations.append(relationship)
        return relations

   
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
