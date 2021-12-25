import os
import re
import pandas as pd

data_path = '../../data/I2B2'
train_path_txt =  os.path.join(data_path, 'concept_assertion_relation_training_data/beth_partners/txt')
train_path_rel = os.path.join(data_path, 'concept_assertion_relation_training_data/beth_partners/rel')
train_path_concept = os.path.join(data_path, 'concept_assertion_relation_training_data/beth_partners/concept')
test_path_txt = os.path.join(data_path, 'reference_standard_for_test_data/txt')
test_path_rel = os.path.join(data_path, 'reference_standard_for_test_data/rel')
test_path_concept = os.path.join(data_path, 'reference_standard_for_test_data/concept')


def get_types(concept_pathes:list):
    """
    :param concept_pathes: list of datasets; train, dev, test
    :return: type mappings
    """
    entitie2type = {}
    for concept_path in concept_pathes:
        for con_path in os.listdir(concept_path):
            with open(os.path.join(concept_path, con_path), 'r', encoding='windows-1252') as fl:
                for con in fl.readlines():
                    con_raw, type_raw = con.rstrip().split('||')
                    con =  re.sub(r'"\s+.+', '', con_raw.replace('c="', ''))
                    type = re.sub(r'"\s+.+', '', type_raw.replace('t="', ''))
                    entitie2type[con] = type
    return entitie2type


def get_data(tx_path, rl_path):
    """
    :param tx_path:
    :param rl_path:
    :return: dataframe containing required fields for training
    """
    entitie2type = get_types([train_path_concept, test_path_concept])
    full_sents, lbls, srcs, trgs, src_types, trg_types = [], [], [], [], [], []
    id2txt = {}
    for txt_path in os.listdir(tx_path):
        with open(os.path.join(tx_path, txt_path), 'r', encoding='windows-1252') as fl:
            id2txt[txt_path.split('.')[0]] = fl.readlines()

    for rel_path in os.listdir(rl_path):
        with open(os.path.join(rl_path, rel_path), 'r', encoding='windows-1252') as fl:
            texts = id2txt[rel_path.split('.')[0]]
            for asrt in fl.readlines():
                subj_raw, rel_raw, obj_raw = asrt.rstrip().split('||')
                subj =  re.sub(r'"\s+.+', '', subj_raw.replace('c="', ''))
                obj = re.sub(r'"\s+.+', '', obj_raw.replace('c="', ''))
                rel = rel_raw.replace('r="', '').replace('"', '')
                for txt in texts:
                    if subj in txt and obj in txt:
                        if not rel in ['TrWP', 'TrIP', 'TrNAP']:
                            full_sents.append(txt.rstrip())
                            srcs.append(subj)
                            trgs.append(obj)
                            src_types.append(entitie2type[subj])
                            trg_types.append(entitie2type[obj])
                            lbls.append(rel)
    df_dict = {'text': full_sents, 'label': lbls, 'srcs': srcs, 'trgs': trgs, 'src_types': src_types, 'trg_types': trg_types}
    return pd.DataFrame(df_dict)


######################################################   Test   ######################################################
if __name__=="__main__":
    test_df = get_data(train_path_txt, train_path_rel)
    print(test_df.shape)
    train_df = get_data(test_path_txt, test_path_rel)
    print(train_df.shape)

