import numpy as np
import spacy
from datasets import load_dataset
import os
from typing import Dict


################# define some useful constants ################
MALE_PRONOUNS = ['he', 'his', 'him', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']
PLURAL_PRONOUNS = ['they', 'them', 'theirs', 'themselves']

# these should be PROPN in order to be detected as a cue
MALE_HONORIFCS = ['master', 'mister', 'mr', 'sir', 'sire', 'gentleman', 'lord', 'esq', 'esquire', 'fr']
FEMALE_HONORIFCS = ['miss', 'ms', 'miss', 'mrs', 'mistress', 'madam', 'maam', 'dame']

# these should be NOUN in order to be detected as a cue
MALE_NONPOS_NAMES = ['man' , 'husband', 'actor', 'prince', 'waiter', 'king', 'businessman', 'sportsman', 'nobleman', 'chairman', 'assemblyman', 'committeeman', 'congressman', 'spokesman', 'batsman', 'alderman', 'anchorman', 'churchman', 'councilman', 'frontman', 'horseman']
FEMALE_NONPOS_NAMES = ['woman', 'wife', 'actress', 'princess', 'waitress', 'queen', 'businesswoman', 'sportswoman', 'noblewoman', 'chairwoman', 'assemblywoman', 'committeewoman', 'congresswoman', 'spokeswoman', 'batswoman', 'alderwoman', 'anchorwoman', 'churchwoman', 'councilwoman', 'frontwoman', 'horsewoman']

# these should become before 'of' in order to be detected as a cue
MALE_POS_NAMES = ['father', 'dad', 'brother', 'nephew', 'boy', 'uncle', 'son', 'grandfather', 'grandson']
FEMALE_POS_NAMES = ['mother', 'mom', 'sister', 'niece', 'girl', 'aunt', 'daughter', 'grandmother', 'granddaughter']

# we will filter any example that has at least on of these occurances
EXCLUDES = ['also known', 'better known', 'ive jerolimov']

MIN_PRONOUN = 2
MAX_PRONOUN = 6
###############################################################



# download test split of wiki_bio dataset from Huggingface Datsets
wiki_bio_dataset = load_dataset('wiki_bio', split='test')

# initialize SpaCy for English
nlp = spacy.load("en_core_web_sm")


# we will just keep the examples that has exactly two parts names + text starts with this name
def name_filter(example:Dict) -> Dict:
  for exclude in EXCLUDES:
    if exclude in example['target_text']:
      return False

  name = None
  if 'name' in example['input_text']['table']['column_header']:
    idx = example['input_text']['table']['column_header'].index('name')
    name = example['input_text']['table']['content'][idx]
    if '\n' in name: name = name[:-1]
    if ('-' in name or len(name.split()) != 2) or name not in example['target_text']:
      return False

  full_name = None
  if 'full_name' in example['input_text']['table']['column_header']:
    idx = example['input_text']['table']['column_header'].index('full_name')
    full_name = example['input_text']['table']['content'][idx]
    if '\n' in full_name: full_name = full_name[:-1]
    if ('-' in full_name or len(full_name.split()) != 2) or full_name not in example['target_text']:
      return False      

  article_title = None
  if 'article_title' in example['input_text']['table']['column_header']:
    idx = example['input_text']['table']['column_header'].index('article_title')
    article_title = example['input_text']['table']['content'][idx]
    if '\n' in article_title: article_title = article_title[:-1]
    if ('-' in article_title or len(article_title.split()) != 2) or article_title not in example['target_text']:
      return False     

  birth_name = None
  if 'birth_name' in example['input_text']['table']['column_header']:
    idx = example['input_text']['table']['column_header'].index('birth_name')
    birth_name = example['input_text']['table']['content'][idx]
    if '\n' in birth_name: birth_name = birth_name[:-1]
    if ('-' in birth_name or len(birth_name.split()) != 2) or birth_name not in example['target_text']:
      return False     

  for i in [name, full_name, article_title, birth_name]:
    for j in [name, full_name, article_title, birth_name]:
      if i and j:
        if i != j: 
          return False

  name = name or full_name or article_title or birth_name
  if name == None or ' '.join(example['target_text'].split()[:2]) != name:
    return False

  return True




# specify gender of the examples
def specify_gender(example:Dict) -> Dict:
#   doc = nlp(example['target_text'])
#   tokens = [token.text for token in doc]
  tokens = example['target_text'].split()
    
  if any(elem in tokens for elem in MALE_PRONOUNS) and any(elem in tokens for elem in FEMALE_PRONOUNS):
    example['gender'] = 'both'
  elif any(elem in tokens for elem in PLURAL_PRONOUNS):
    example['gender'] = 'plural'
  elif any(elem in tokens for elem in MALE_PRONOUNS):
    example['gender'] = 'male'
  elif any(elem in tokens for elem in FEMALE_PRONOUNS):
    example['gender'] = 'female'
  else:
    example['gender'] = None

  return example




# we will just use texts that are either male or female + has between MIN_PRONOUN to MAX_PRONOUN pronouns
def num_pronouns_filter(example:Dict) -> Dict:
  if example['gender'] == 'male':
    # doc = nlp(example['target_text'])
    # tokens = [token.text for token in doc]
    tokens = example['target_text'].split()
    num_pronouns = np.array([token in MALE_PRONOUNS for token in tokens]).sum()
    if MIN_PRONOUN <= num_pronouns <= MAX_PRONOUN:
      return True

  if example['gender'] == 'female':
    # doc = nlp(example['target_text'])
    # tokens = [token.text for token in doc]
    tokens = example['target_text'].split()
    num_pronouns = np.array([token in FEMALE_PRONOUNS for token in tokens]).sum()
    if MIN_PRONOUN <= num_pronouns <= MAX_PRONOUN:
      return True

  return False




# CREATE DATASET PRCEDURE
def create_dataset(example:Dict) -> Dict:
  if example['gender'] == 'male':
    prounouns = MALE_PRONOUNS
    honorifics = MALE_HONORIFCS
    non_pos_names = MALE_NONPOS_NAMES
    pos_names = MALE_POS_NAMES
  else:
    prounouns = FEMALE_PRONOUNS
    honorifics = FEMALE_HONORIFCS
    non_pos_names = FEMALE_NONPOS_NAMES
    pos_names = FEMALE_POS_NAMES   


  # init a list to add found cues to it
  all_cues = []
  # process the text using SpaCy
  doc = nlp(example['target_text'])

  # obtain first_name using a simple heurestic
  tokens = [token.text for token in doc]

  first_name = tokens[0]
  last_name = tokens[1]

  # iterate over all tokens of the text and check for possible cues
  for token in doc:
    # first get the span of this token in the text
    start, end = token.idx, token.idx + len(token)

    # check whether it is a first_name
    if token.text == first_name:
      all_cues.append(f'{token.text}|{start}|{end}')  

    # check whether it is a last_name
    elif token.text == last_name:
      all_cues.append(f'{token.text}|{start}|{end}')  

    # check whether is it a honorific name
    elif token.text in honorifics and token.pos_ == 'PROPN':
      all_cues.append(f'{token.text}|{start}|{end}')  

    # check whether is it a non_possessional name
    elif token.text in non_pos_names and token.pos_ == 'NOUN':
      all_cues.append(f'{token.text}|{start}|{end}') 

    # check whether is it a possessional name
    elif token.text in pos_names and example['target_text'][end+2:end+4] == 'of':
      all_cues.append(f'{token.text}|{start}|{end}') 

    # check whether is it a pronoun
    elif token.text in prounouns and token.pos_ == 'PRON':
      all_cues.append(f'{token.text}|{start}|{end}') 
      

  # find the last pronoun of the text and mask it
  for cue in reversed(all_cues):
    token, start, end = cue.split('|')
    if token in prounouns:
      mask_word, mask_start_idx = token, int(start)
      break

  masked_text = example['target_text'][:mask_start_idx]
  masked_text += '[MASK]'

  # cue words of the masked_text should be before the masked_word
  masked_text_cues = []
  for cue in all_cues:
    token, start, end = cue.split('|')
    if int(start) < mask_start_idx:
      masked_text_cues.append(cue)

  example['masked_text'] = masked_text
  example['target_word'] = mask_word
  example['cue_words'] = masked_text_cues

  return example




wiki_bio_dataset_filtered = wiki_bio_dataset.filter(name_filter)
updated_dataset = wiki_bio_dataset_filtered.map(specify_gender)
updated_dataset_filtered = updated_dataset.filter(num_pronouns_filter)
gender_agreement_dataset = updated_dataset_filtered.map(create_dataset)


# save the resulting dataset as a json file
DATA_DIR = 'data/'
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
gender_agreement_dataset.to_json(DATA_DIR + 'gender_agreement.json') 
