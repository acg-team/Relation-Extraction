import os
import pandas as pd
from nltk import sent_tokenize
from collections import defaultdict
import csv

data_path = '../../data/ChemProt'
train_path =  os.path.join(data_path, 'chemprot_training')
dev_path =  os.path.join(data_path, 'chemprot_development')
test_path =  os.path.join(data_path, 'chemprot_test_gs')

train_entities = os.path.join(train_path, 'chemprot_training_entities.tsv')
dev_entities = os.path.join(dev_path, 'chemprot_development_entities.tsv')
test_entities = os.path.join(test_path, 'chemprot_test_entities_gs.tsv')


def get_types(concept_pathes:list):
    """
    :param concept_pathes: list of datasets; train, dev, test
    :return: type mappings
    """
    entitie2type = defaultdict(dict)
    for entity_path in concept_pathes:
        with open(entity_path) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for line in tsvreader:
                entitie2type[line[0]][line[1]] = (line[5], line[2], [int(line[3]), int(line[4])])

    return entitie2type


def get_data(abs_path, rl_path):
    """
    :param abs_path:
    :param rl_path:
    :return: dataframe containing required fields for training
    """
    entitie2type = get_types([train_entities, dev_entities, test_entities])

    full_sents, between_sents, lbls, srcs, trgs, src_types, trg_types = [], [], [], [], [], [], []
    id2abstracts = {}
    with open(abs_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            id2abstracts[line[0]] = ([line[1]] + sent_tokenize(line[2]), line[1]+' '+line[2])

    with open(rl_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            pmid = line[0]
            sentences_list = id2abstracts[pmid][0]
            abstract_whole = id2abstracts[pmid][1]
            sub = line[4].replace('Arg1:', '')
            obj = line[5].replace('Arg2:', '')
            lbl = line[1]
            if lbl not in ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9']:continue
            sub_txt = entitie2type[pmid][sub][0]
            sub_type = entitie2type[pmid][sub][1]
            obj_txt = entitie2type[pmid][obj][0]
            obj_type = entitie2type[pmid][obj][1]

            if entitie2type[pmid][sub][2][0] < entitie2type[pmid][obj][2][0]:
                between_sent = abstract_whole[entitie2type[pmid][sub][2][0]:entitie2type[pmid][obj][2][1]]
            else:
                between_sent = abstract_whole[entitie2type[pmid][obj][2][0]:entitie2type[pmid][sub][2][1]]

            full_sent = between_sent# for 44 cases due to sentnce spitter error (e.g.) there is not full matching sentnce
            for sent in sentences_list:
                if between_sent in sent:
                    full_sent = sent
                    break

            full_sents.append(full_sent)
            between_sents.append(between_sent)
            lbls.append(lbl)
            srcs.append(sub_txt)
            trgs.append(obj_txt)
            src_types.append(sub_type)
            trg_types.append(obj_type)

    df_dict = {'text': full_sents, 'between_text': between_sents, 'label': lbls, 'srcs': srcs, 'trgs': trgs,
               'src_types': src_types, 'trg_types': trg_types}
    df = pd.DataFrame(df_dict)
    return df.sample(frac=1).reset_index(drop=True)


######################################################   Test   ######################################################
if __name__=="__main__":
    df_dict_train = get_data(os.path.join(train_path, 'chemprot_training_abstracts.tsv'), os.path.join(train_path, 'chemprot_training_relations.tsv'))
    df_dict_dev = get_data(os.path.join(dev_path, 'chemprot_development_abstracts.tsv'), os.path.join(dev_path, 'chemprot_development_relations.tsv'))
    df_dict_test = get_data(os.path.join(test_path, 'chemprot_test_abstracts_gs.tsv'), os.path.join(test_path, 'chemprot_test_relations_gs.tsv'))

    print(df_dict_train.shape)
    print(df_dict_dev.shape)
    print(df_dict_test.shape)


