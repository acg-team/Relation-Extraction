import os
import json
import string
import pandas as pd
from nltk import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, OrderedDict

data_path = '../../data/AGAC'

train_path = os.path.join(data_path, 'AGAC_training/json')
test_path = os.path.join(data_path, 'AGAC_test/json')
AGAC_TASK3 = os.path.join(data_path, 'AGAC_TASK3.csv')


def traverse_folder(base_folder: str, file_format: str ='.json' ) -> str:

    """
    :param base_folder:
    :return: iteratively go through all sub folders and visit their xml files
    """

    for root, dirs, files in os.walk(base_folder):
        for name in files:
            if name.endswith(file_format):
                yield os.path.join(root, name)


def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def get_data(data_path):
    """
    :param data_path:
    :return: dataframe containing required fields for training
    """
    task3_df = pd.read_csv(AGAC_TASK3)
    pm_id2AGAC_task3 = defaultdict(list)
    for i in zip(task3_df['PMID'], task3_df['GENE'], task3_df['FUNCTION'], task3_df['DISEASE']):
        pm_id2AGAC_task3[str(i[0])].append([x.lower() for x in i[1:]])
    pm_id2AGAC_task3 = OrderedDict(sorted(pm_id2AGAC_task3.items()))

    sentences = []
    t3_sents, t3_rel, t3_srcs, t3_trgs, t3_src_types, t3_trg_types = [], [], [], [], [], []
    ordered_pathes = list(traverse_folder(data_path, '.json'))
    ordered_pathes.sort()
    entity2type = {}
    for file_path in ordered_pathes:
        json_obj = json.load(open(file_path, 'r'))
        full_text = json_obj['text'].lower()

        entity_id = {}
        for item in json_obj['denotations']:
            first_char = int(item['span']['begin'])
            last_char = int(item['span']['end'])
            entity = full_text[first_char:last_char]

            entity_type = item['obj']
            entity_id[item['id']] = {'entity': entity, 'type': entity_type, 'span': item['span']}
            entity2type[entity] = entity_type

        for item in json_obj['relations']:
            source = entity_id[item['subj']]
            target = entity_id[item['obj']]

            source_entity = clean_doc(source['entity'])
            target_entity = clean_doc(target['entity'])

            source_type = source['type']
            target_type = target['type']

            span = clean_doc(full_text[int(source['span']['begin']):int(target['span']['end'])])
            rel = item['pred']
            sentences.append({'text': clean_doc(full_text), 'source_entity': source_entity, 'target_entity': target_entity,
                              'source_type': source_type, 'target_type': target_type,'span':span, 'rel':rel})

    #############################################   TASK 3   #############################################
    sub_obj2sent = defaultdict(list)
    all_sentences = set()
    for file_path in ordered_pathes:
        json_obj = json.load(open(file_path, 'r'))
        full_text = json_obj['text'].lower()

        sourceid = json_obj['sourceid']
        if sourceid in pm_id2AGAC_task3:
            for rel in pm_id2AGAC_task3[sourceid]:
                sub, pred, obj = rel
                for sent in sent_tokenize(full_text):
                    all_sentences.add(sent)
                    if sub in sent and obj in sent:
                        sub_obj2sent[(sub, obj)].append(sent)
                        t3_rel.append(pred)
                        t3_sents.append(sent)
                        t3_srcs.append(sub)
                        t3_trgs.append(obj)
                        t3_src_types.append(entity2type.get(sub, 'None'))
                        t3_trg_types.append(entity2type.get(obj, 'None'))

    t3_dict = {'text': t3_sents, 'label': t3_rel, 'srcs': t3_srcs, 'trgs': t3_trgs, 'src_types': t3_src_types, 'trg_types': t3_trg_types}
    return pd.DataFrame(t3_dict)


######################################################   Test   ######################################################
if __name__=="__main__":
    t2_df, t3_df= get_data(train_path)
    print(t3_df.shape)

    print(t2_df.shape)
    for i in list(zip(t2_df['src_types'], t2_df['trg_types'])):
        print(i)
    for i in t2_df['src_types']:
        print(i)

