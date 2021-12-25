import os
import xml.etree.ElementTree as ET
import pandas as pd


data_path = '../../data/DDI'
train_path = os.path.join(data_path, 'Train')
test_path = os.path.join(data_path, 'Test/Test for DDI Extraction task')


def get_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                yield file_path


def get_types(data_pathes:list):
    """
    :param data_pathes: list of datasets; train, dev, test
    :return: type mappings
    """
    entitie2type = {}
    for data_path in data_pathes:
        for file_path in get_files(data_path):
            root = ET.parse(file_path).getroot()
            for sentence in root.findall("sentence"):
                for entity in sentence.findall("entity"):
                    ent_type = entity.attrib['type']
                    ent_text = entity.attrib['text']
                    entitie2type[ent_text] = ent_type
    return entitie2type


entitie2type = get_types([train_path, test_path])

def get_data(data_path):
    """
    :param data_path:
    :return: dataframe containing required fields for training
    """

    sentences = []

    for file_path in get_files(data_path):
        root = ET.parse(file_path).getroot()
        for sentence in root.findall("sentence"):
            ent_id2text_position = {}
            sent_text = sentence.attrib['text']
            pairs = []
            for entity in sentence.findall("entity"):
                ent_text = entity.attrib['text']
                charOffset = entity.attrib['charOffset']
                ent_id2text_position[entity.attrib['id']] = (ent_text, charOffset)

            for entity in sentence.findall("pair"):
                pair_relation = entity.attrib['ddi']

                if pair_relation == 'true':
                    try:
                        pair_type = entity.attrib['type']
                        pair_source, source_offset = ent_id2text_position[entity.attrib['e1']]
                        pair_target, target_offset = ent_id2text_position[entity.attrib['e2']]

                        # 53-59 117-123;157-160: if offset has more than one entity assuming first entity
                        src_idx = int(source_offset.split(';')[0].split('-')[0])
                        # if target entity has more than one position assume last entity
                        trg_idx = int(target_offset.split(';')[-1].split('-')[1])
                        pairs.append((pair_source, pair_target, pair_type, sent_text[src_idx:trg_idx]))
                    except KeyError:
                        print('True but no type!')
            sentences.append({'text':sent_text, 'pairs':pairs})

    full_sents, between_sents, lbls, srcs, trgs, src_types, trg_types = [], [], [], [], [], [], []

    for s in sentences:
        for p in s['pairs']:
            full_sents.append(s['text'])
            between_sents.append(p[3])
            lbls.append(p[2])
            srcs.append(p[0])
            trgs.append(p[1])
            src_types.append(entitie2type[p[0]])
            trg_types.append(entitie2type[p[1]])

    df_dict = {'text': full_sents, 'between_sents': between_sents, 'label': lbls, 'srcs': srcs, 'trgs': trgs,
               'src_types': src_types, 'trg_types': trg_types}
    return pd.DataFrame(df_dict)


######################################################   Test   ######################################################
if __name__=="__main__":
    test_df = get_data(test_path)
    print(test_df.shape)
    train_df = get_data(train_path)
    print(train_df.shape)
