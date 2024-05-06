import numpy as np
import spacy
from datasets import load_dataset
from tqdm import tqdm
import random
import json
import os
from typing import List, Tuple, Dict


################# define some useful constants ################
MALE_PRONOUNS = ['he', 'his', 'him']
FEMALE_PRONOUNS = ['she', 'her', 'hers']
PLURAL_PRONOUNS = ['they', 'them', 'theirs']

# these should be PROPN in order to be detected as a cue
MALE_HONORIFCS = ['master', 'mr', 'sir', 'sire', 'gentleman', 'lord', 'esq', 'fr']
FEMALE_HONORIFCS = ['miss', 'ms', 'miss', 'mrs', 'mistress', 'madam', 'maam', 'dame']

# these should be NOUN in order to be detected as a cue
MALE_NONPOS_NAMES = ['man' , 'husband', 'actor', 'prince', 'waiter', 'king', 'grandfather']
FEMALE_NONPOS_NAMES = ['woman', 'wife', 'actress', 'princess', 'waitress', 'qeen', 'grandmother']

# these should become after 'the' in order to be detected as a cue
MALE_POS_NAMES = ['father', 'boy', 'uncle', 'son']
FEMALE_POS_NAMES = ['mother', 'girl', 'aunt', 'daughter']

MAX_PRONOUN = 6
MALE_DATSET_SIZE = 5000
FEMALE_DATSET_SIZE = 5000
###############################################################



# download wiki_bio dataset from Huggingface Datsets
wiki_bio_dataset = load_dataset('wiki_bio')


nlp = spacy.load("en_core_web_sm")


# we will just use texts that has between 1 to MAX_PRONOUN pronouns
def find_num_pronouns(examples: List[str], prounouns: List[str]) -> List[Tuple[str, int]]:
    output_aug = []
    for text in examples:
        words = text.split()
        num_pronouns = np.array([word in prounouns for word in words]).sum()
        if 1 <= num_pronouns <= MAX_PRONOUN:
            output_aug.append((text, num_pronouns))
    return output_aug


# CREATE DATASET PRCEDURE
def create_dataset(examples_aug: List[Tuple[str, int]], prounouns: List[str], honorifics: List[str],
                   non_pos_names: List[str], pos_names: List[str]) -> List[Dict]:
    dataset = []
    for text, num_pronouns in tqdm(examples_aug):
        # init an entry for this datapoint to be place in dataset dict
        entry = {}
        # init a  dict to add found cues to it
        complete_cues = {}
        # process the text using SpaCy
        doc = nlp(text)

        # obtain first_name using a simple heurestic
        tokens = [token.text for token in doc]
        if tokens[0] not in honorifics:
            first_name = tokens[0]
        else:
            first_name = tokens[1]

        # iterate over all tokens of the text and check for possible cues
        for token in doc:
            # first get the span of this token in the text
            start, end = token.idx, token.idx + len(token)

            # check whether it is a first_name
            if token.text == first_name:
                complete_cues[token.text] = (start, end)

            # check whether is it a honorific name
            elif token.text in honorifics and token.pos_ == 'PROPN':
                complete_cues[token.text] = (start, end)

            # check whether is it a non_possessional name
            elif token.text in non_pos_names and token.pos_ == 'NOUN':
                complete_cues[token.text] = (start, end)

            # check whether is it a possessional name
            elif token.text in pos_names and text[start-4:start-1] == 'the':
                complete_cues[token.text] = (start, end)

            # check whether is it a pronoun
            elif token.text in prounouns and token.pos_ == 'PRON':
                complete_cues[token.text] = (start, end)
            

        # find the last pronoun  of the text and mask it
        for key, value in reversed(complete_cues.items()):
            if key in prounouns:
                mask_word, mask_indices = key, value
                break

        masked_text = text[:mask_indices[0]]
        masked_text += '[MASK]'

        # cue words of the masked_text should be before the masked_word
        masked_cues = {}
        for key, value in complete_cues.items():
            if value[0] < mask_indices[0]:
                masked_cues[key] = value

        #  complete the entry of this datapoint
        entry['complete_text'] = text
        entry['masked_text'] = masked_text
        entry['target_word'] = mask_word
        entry['complete_text_cue_words'] = complete_cues
        entry['masked_text_cue_words'] = masked_cues

        # add this entry to the cueWords_dataset
        dataset.append(entry)

    return dataset



# extract texts of the wiki_bio dataset from it's TEST split
texts = []
for data in  wiki_bio_dataset['test']:
  texts.append(data['target_text'])


# devide the texts based on which kind of genders are present in that.
# we will just use single gender involved texts
double_genders = []
plural_genders = []
male_genders = []
female_genders = []

for text in texts:
    if any(elem in text.lower().split() for elem in MALE_PRONOUNS) and any(elem in text.lower().split() for elem in FEMALE_PRONOUNS):
        double_genders.append(text)
    elif any(elem in text.lower().split() for elem in PLURAL_PRONOUNS):
        plural_genders.append(text)
    elif any(elem in text.lower().split() for elem in MALE_PRONOUNS):
        male_genders.append(text)
    elif any(elem in text.lower().split() for elem in FEMALE_PRONOUNS):
        female_genders.append(text)


male_genders_aug = find_num_pronouns(male_genders, MALE_PRONOUNS)
female_genders_aug = find_num_pronouns(female_genders, FEMALE_PRONOUNS)


random.shuffle(male_genders_aug)
if len(male_genders_aug) > MALE_DATSET_SIZE:
    male_genders_aug = male_genders_aug[:MALE_DATSET_SIZE]
else:
    raise Warning()

random.shuffle(female_genders_aug)
if len(female_genders_aug) > FEMALE_DATSET_SIZE:
    female_genders_aug = female_genders_aug[:FEMALE_DATSET_SIZE]
else:
    raise Warning()


male_dataset = create_dataset(male_genders_aug, MALE_PRONOUNS, MALE_HONORIFCS, MALE_NONPOS_NAMES, MALE_POS_NAMES)
female_dataset = create_dataset(female_genders_aug, FEMALE_PRONOUNS, FEMALE_HONORIFCS, FEMALE_NONPOS_NAMES, FEMALE_POS_NAMES)

dataset = male_dataset + female_dataset

# save it as a json file
json_data = json.dumps(dataset, indent=4)
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
with open(data_dir + 'gender_agreement.json', "w") as f:
    f.write(json_data)
