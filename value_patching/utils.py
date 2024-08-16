# Note: This script currently runs only with batch_size equals to 1

from modeling_roberta_forVP import RobertaForMaskedLM
from modeling_bert_forVP import BertForMaskedLM
from modeling_gpt2_forVP import GPT2LMHeadModel
from modeling_gemma_forVP import GemmaForCausalLM
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm
import random
from IPython.display import display
import numpy as np
import torch
from typing import List, Tuple
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from huggingface_hub import notebook_login
import copy
from collections import defaultdict


MALE_PRONOUNS = ['he', 'his', 'him', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']
PLURAL_PRONOUNS = ['they', 'them', 'theirs', 'themselves']

# these should be PROPN in order to be detected as a cue
MALE_HONORIFCS = ['master', 'mister', 'mr', 'sir', 'sire', 'gentleman', 'lord']
FEMALE_HONORIFCS = ['miss', 'ms', 'mrs', 'mistress', 'madam', 'maam', 'dame']

# these should be NOUN in order to be detected as a cue
MALE_NONPOS_NAMES = ['man' , 'actor', 'prince', 'waiter', 'king', 'businessman', 'sportsman', 'nobleman', 'chairman', 'assemblyman', 'committeeman', 'congressman', 'spokesman', 'batsman', 'alderman', 'anchorman', 'churchman', 'councilman', 'frontman', 'horseman']
FEMALE_NONPOS_NAMES = ['woman', 'actress', 'princess', 'waitress', 'queen', 'businesswoman', 'sportswoman', 'noblewoman', 'chairwoman', 'assemblywoman', 'committeewoman', 'congresswoman', 'spokeswoman', 'batswoman', 'alderwoman', 'anchorwoman', 'churchwoman', 'councilwoman', 'frontwoman', 'horsewoman']

# these should come before 'of' in order to be detected as a cue
MALE_POS_NAMES = ['father', 'dad', 'husband', 'brother', 'nephew', 'boy', 'uncle', 'son', 'grandfather', 'grandson']
FEMALE_POS_NAMES = ['mother', 'mom', 'wife', 'sister', 'niece', 'girl', 'aunt', 'daughter', 'grandmother', 'granddaughter']


class dataset2VPs:
    def __init__(
            self,
            model_checkpoint: str,
            dataset_path: str,
            num_cues: int
    ) -> pd.DataFrame:
        self.model_checkpoint = model_checkpoint
        self.dataset_path = dataset_path
        self.num_cues = num_cues

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        # Add a padding token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _get_token_idxes_for_cues(self, sample) -> List[List[Tuple[int, int]]]:
        """
        Returns start and end indices of tokens corresponding to each cue word.
        Note that indices are based on input_ids list.
        """
        text = sample['masked_text']
        encoded = self.tokenizer.encode_plus(text, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]

        cues_tokenIdxes = []
        for cue_word in sample['cue_words']:
            cue, cue_start, cue_end = cue_word.split('|')
            cue_start, cue_end = int(cue_start), int(cue_end)

            if 'gemma' in self.model_checkpoint or 'gpt2' in self.model_checkpoint:
              if cue_start > 0:
                if text[cue_start - 1] == ' ':
                  cue_start -= 1

            word_token_positions = []
            for i, (start, end) in enumerate(offsets):
                if cue_start <= start <= cue_end and cue_start <= end <= cue_end and start < end:
                    word_token_positions.append(i)
            cues_tokenIdxes.append([word_token_positions[0], word_token_positions[-1]+1])

        return  cues_tokenIdxes
    

    def _make_ablation(self, example):
        replace = 'he' if example['gender'] == 'male' else 'she'
        tokens = example['masked_text'].split()
        replace_tokens = [replace] + tokens[2:]
        example['masked_text'] = ' '.join(replace_tokens)

        replace_cue = 'he|0|2' if example['gender'] == 'male' else 'she|0|3'
        replace_cue_size = 2 if example['gender'] == 'male' else 3

        _, _, lastName_cue_end = example['cue_words'][1].split('|')
        example['cue_words'] = [replace_cue] + example['cue_words'][2:]

        for i, cue_word in enumerate(example['cue_words']):
            if i > 0:
                cue, cue_start, cue_end = cue_word.split('|')
                cue_start, cue_end = int(cue_start), int(cue_end)
                cue_start -= (int(lastName_cue_end) - replace_cue_size)
                cue_end -= (int(lastName_cue_end) - replace_cue_size)
                example['cue_words'][i] = cue + '|' + str(cue_start) + '|' + str(cue_end)

        example['cues_pattern'] = 'P' + example['cues_pattern'][2:]
        return example
    

    def _suitable_mask(self, example):
        # the format is alread suitable for BERT since we used [MASK] token when constructing our dataset
        if 'roberta' in self.model_checkpoint:
            example['masked_text'] = example['masked_text'].replace('[MASK]', '<mask>')
        elif 'gemma' in self.model_checkpoint or 'gpt2' in self.model_checkpoint:
            mask_start_idx = example['masked_text'].find(' [MASK]')
            if mask_start_idx == -1:
                mask_start_idx = example['masked_text'].find('[MASK]')
            example['masked_text'] = example['masked_text'][:mask_start_idx]
        return example


    def _preprocess_function(self, examples):
        # Tokenize the texts
        args = (examples['masked_text'],)
        result = self.tokenizer(*args, padding=False, truncation=False)
        return result


    def _check_input_length_wrapped(self, model):
        def check_input_length(example):
            config = model.config
            if 'gpt2' in self.model_checkpoint:
                max_input_length = config.n_positions
            else:
                max_input_length = config.max_position_embeddings 
            if len(example['input_ids']) > max_input_length:
                return False
            return True
        return check_input_length
    

    def _excludeForCorrupt(self, example):
        # idxes_to_check = []
        # first_name, last_name = example['masked_text'].split()[:2]      
        # for i, cue_word in enumerate(example['cue_words']):
        #     cue, cue_start, cue_end = cue_word.split('|')
        #     if cue == first_name or cue == last_name:
        #         idxes_to_check.append(i)
        
        # max_allow = 2 if 'roberta' in self.model_checkpoint else 1 
        max_allow = 2
        for i in range(len(example['cues_pattern'])):
            if example['cues_pattern'][i] in ['F', 'L']:
                if example['cues_tokenIdxes'][i][1] - example['cues_tokenIdxes'][i][0] > max_allow:
                    # print(example)
                    # print("#"*100)
                    return False
        
        inconsistant_num_tokens = ['businessman', 'businesswoman', 'sportsman', 'sportswoman', 'chairman', 
                                   'chairwoman', 'committeeman', 'committeewoman', 'congressman', 'congresswoman', 
                                   'frontman', 'frontwoman']
        for inconsistant_num_token in inconsistant_num_tokens:
            if inconsistant_num_token in example['masked_text']:
                return False
            
        return True
    

    def _get_corrupt_name(self, gender, cue_id, firstName):
        num_tokens = cue_id[1] - cue_id[0]
        if firstName:
                if num_tokens == 1:
                    male_replace_name, female_replace_name = ' bob', ' amy'
                    male_replace_id = self.tokenizer.encode(male_replace_name, add_special_tokens=False)
                    female_replace_id = self.tokenizer.encode(female_replace_name, add_special_tokens=False)
                    return female_replace_id if gender == 'male' else male_replace_id
                elif num_tokens == 2:
                    male_replace_name, female_replace_name = ' aron', ' noora'
                    male_replace_id = self.tokenizer.encode(male_replace_name, add_special_tokens=False)
                    female_replace_id = self.tokenizer.encode(female_replace_name, add_special_tokens=False)
                    return female_replace_id if gender == 'male' else male_replace_id
        
        else:
            if num_tokens == 1:
                replace_name = 'walker'
                replace_id = self.tokenizer.encode(replace_name, add_special_tokens=False)
                return replace_id
            elif num_tokens == 2:
                replace_name = ' willinsky'
                replace_id = self.tokenizer.encode(replace_name, add_special_tokens=False)
                return replace_id           


    def _add_corrupt(self, example):
        # text_clean = example['masked_text']
        # first_name, last_name = text_clean.split()[:2]
        # print(example['cues_tokenIdxes'])
        # print(example['cue_words'], first_name, last_name)
        cue_words_corrupt_tokenIdxes = []

        cue_dict = {
            'male': (MALE_PRONOUNS, FEMALE_PRONOUNS, MALE_HONORIFCS, FEMALE_HONORIFCS, MALE_NONPOS_NAMES, FEMALE_NONPOS_NAMES, MALE_POS_NAMES, FEMALE_POS_NAMES),
            'female': (FEMALE_PRONOUNS, MALE_PRONOUNS, FEMALE_HONORIFCS, MALE_HONORIFCS, FEMALE_NONPOS_NAMES, MALE_NONPOS_NAMES, FEMALE_POS_NAMES, MALE_POS_NAMES)
        }

        pronouns, opposite_pronouns, honorifics, opposite_honorifics, nonpos_names, opposite_nonpos_names, pos_names, opposite_pos_names = cue_dict[example['gender']]

        for i, (cue_word, cue_tokenId) in enumerate(zip(example['cue_words'], example['cues_tokenIdxes'])):
            cue, cue_start, cue_end = cue_word.split('|')
            cue_start, cue_end = int(cue_start), int(cue_end)

            if example['cues_pattern'][i] == 'F':
                cue_words_corrupt_tokenIdxes.append(self._get_corrupt_name(example['gender'], cue_tokenId, firstName=True))
            elif example['cues_pattern'][i] == 'L':
                cue_words_corrupt_tokenIdxes.append(self._get_corrupt_name(example['gender'], cue_tokenId, firstName=False))
            elif cue in pronouns:
                idx = pronouns.index(cue)
                cue_words_corrupt_tokenIdxes.append(self.tokenizer.encode(' ' + opposite_pronouns[idx], add_special_tokens=False))
            elif cue in honorifics:
                idx = honorifics.index(cue)
                cue_words_corrupt_tokenIdxes.append(self.tokenizer.encode(' ' + opposite_honorifics[idx], add_special_tokens=False))
            elif cue in nonpos_names:
                idx = nonpos_names.index(cue)
                cue_words_corrupt_tokenIdxes.append(self.tokenizer.encode(' ' + opposite_nonpos_names[idx], add_special_tokens=False))
            elif cue in pos_names:
                idx = pos_names.index(cue)
                cue_words_corrupt_tokenIdxes.append(self.tokenizer.encode(' ' + opposite_pos_names[idx], add_special_tokens=False))

        example['input_ids_corrupt'] = example['input_ids'].copy()
        for i, cues_tokenId in enumerate(example['cues_tokenIdxes']):
            example['input_ids_corrupt'][cues_tokenId[0]:cues_tokenId[1]] = cue_words_corrupt_tokenIdxes[i]

        return example


    def _extract_pred_words_probs(self, predictions, pos4preds):
        preds_idxes = [torch.topk(predictions[j, pos4preds[j]], 1).indices.tolist()[0] for j in range(len(predictions))]
        preds_words = [self.tokenizer.decode([idx]) for idx in preds_idxes]
        preds_probs = [torch.softmax(predictions, dim=0)[j, pos4preds[j], idx].item() for j, idx in enumerate(preds_idxes)]
        
        return preds_words, preds_probs


    # def _make_balanced(self, examples):
    #     occurrences = defaultdict(list)
    #     for idx, pattern in enumerate(examples['cues_pattern']):
    #         if (pattern[0]=='F' and pattern[1]=='L') and (pattern[-1] in ['P', 'O']):
    #             occurrences[pattern[-1]].append(idx)

    #     print(occurrences)

    #     min_count = min(len(indices) for indices in occurrences.values())

    #     balanced_indices = []
    #     for indices in occurrences.values():
    #         balanced_indices.extend(indices[:min_count])

    #     mask = [i in balanced_indices for i in range(len(examples['cues_pattern']))]

    #     return {
    #         key: [value for value, keep in zip(values, mask) if keep]
    #         for key, values in examples.items()
    #     }



    def retrieveVP(self, ablation_study: bool = False) -> pd.DataFrame:
        """
        Main function that retrievs activation patching results for the cue words.
        """
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 1

        # set up the model
        if "bert" in self.model_checkpoint:
            model = BertForMaskedLM.from_pretrained(self.model_checkpoint)
        elif "gpt2" in self.model_checkpoint:
            model = GPT2LMHeadModel.from_pretrained(self.model_checkpoint)
        elif "gemma" in self.model_checkpoint:
            model = GemmaForCausalLM.from_pretrained(self.model_checkpoint, attn_implementation='eager')
        elif "roberta" in self.model_checkpoint:
            model = RobertaForMaskedLM.from_pretrained(self.model_checkpoint)
        else:
            print(self.model_checkpoint + ' not implemented!')

        model.eval()

        # load test split of our gender_agreement dataset
        dataset = load_from_disk(self.dataset_path)['test']

        # seperate examples that has the given number of cue words
        sel_dataset = dataset.filter(lambda example: len(example['cue_words'])==self.num_cues)

        # make the dataset size equals to the size of the dataset with 6 cues -> making the dataset balanced
        sel_dataset = sel_dataset.select(range(sel_dataset[0]['balance_size']))

        # replace name to he/she if we are doing the ablation study
        if ablation_study:
            all_cues_tokenIdxes = []
            for i in tqdm(range(len(sel_dataset)), desc="Extracting cue words tokens indices"):
                cues_tokenIdxes = self._get_token_idxes_for_cues(sel_dataset[i])
                all_cues_tokenIdxes.append(cues_tokenIdxes)
            sel_dataset = sel_dataset.add_column("cues_tokenIdxes", all_cues_tokenIdxes)

            # added to make the comparison between cm and vp results fair
            sel_dataset = sel_dataset.filter(self._excludeForCorrupt, batched=False)
            sel_dataset = sel_dataset.map(self._make_ablation, batched=False)

        # each model has it's own mask token
        sel_dataset = sel_dataset.map(self._suitable_mask, batched=False)

        # add tokenization output of each sample to the dataset
        sel_dataset = sel_dataset.map(self._preprocess_function, batched=True, batch_size=1024)

        # ensure that each example is shorter than the model max input length
        sel_dataset = sel_dataset.filter(self._check_input_length_wrapped(model), batched=False)

        all_cues_tokenIdxes = []
        for i in tqdm(range(len(sel_dataset)), desc="Extracting cue words tokens indices"):
            cues_tokenIdxes = self._get_token_idxes_for_cues(sel_dataset[i])
            all_cues_tokenIdxes.append(cues_tokenIdxes)

        if 'cues_tokenIdxes' in sel_dataset.column_names:
            sel_dataset = sel_dataset.remove_columns(['cues_tokenIdxes'])
        sel_dataset = sel_dataset.add_column("cues_tokenIdxes", all_cues_tokenIdxes)

        # some exclusisons before adding a corrupted text
        sel_dataset = sel_dataset.filter(self._excludeForCorrupt, batched=False)

        # add a corrupted text for each example
        sel_dataset = sel_dataset.map(self._add_corrupt, batched=False)

        # # make dataset balanced in cues_pattern
        # sel_dataset = sel_dataset.map(self._make_balanced, batched=True, batch_size=len(sel_dataset))

        # add index of the each example in the dataset 
        sel_dataset = sel_dataset.add_column("idx", [i for i in range(len(sel_dataset))])

        # we will output this final_data as a pandas dataframe
        final_data = {
            "masked_text": [],  # List[str]
            "target_word": [],  # List[str]
            "cue_words": [],    # List[List[str]]
            "cues_pattern": []  # List[str]
        }

        lengths = []
        for i in tqdm(range(len(sel_dataset)), desc="Initializing final data"):

            final_data["masked_text"].append(sel_dataset[i]['masked_text'])
            final_data["target_word"].append(sel_dataset[i]['target_word'])
            final_data["cue_words"].append(sel_dataset[i]['cue_words'])
            final_data["cues_pattern"].append(sel_dataset[i]['cues_pattern'])

            length = len(sel_dataset[i]['input_ids'])
            lengths.append(length)
        
        sel_dataset = sel_dataset.add_column("length", lengths)

        dataset_size = len(sel_dataset)
        steps = int(np.ceil(dataset_size / BATCH_SIZE))

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        cols = ["input_ids", "input_ids_corrupt", "attention_mask", "idx", "length", "cues_tokenIdxes"]
        sel_dataset_final = copy.deepcopy(sel_dataset)
        sel_dataset_final.set_format(type="torch", columns=cols)



        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        sampler = LengthGroupedSampler(
            BATCH_SIZE,
            lengths=lengths,
            model_input_name=self.tokenizer.model_input_names[0],
            generator=generator,
        )


        dataloader = DataLoader(
            sel_dataset_final,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            collate_fn=collator,
        )


        model.to(DEVICE)
        it = iter(dataloader)
        idxes = []

        # we are going to retrieve these values for the target token of each example
        shuffled_data = {
            "input_ids": [],
            "input_ids_corrupt": [],
            "pos4pred": [],  # List[int]

            "model_top1_clean_prediction": [],  # List[str]
            "model_top1_clean_confidence": [],  # List[float]

            "model_top1_corrupt_prediction": [],  # List[str]
            "model_top1_corrupt_confidence": [],  # List[float]

            "vp_all_layersAndPos": [],   # List[tensor(layer, seq_len i.e len of input_ids w/o any padding)]

            "cues_tokenIdxes": [] # List[array(num_cues, 2)]
        }

        with torch.no_grad():
            for i in tqdm(range(steps), desc="Forwarding and extracting value patching scores"):
                batch = next(it)
                if 'roberta' in self.model_checkpoint or 'bert' in self.model_checkpoint:
                    is_encoder = True
                elif 'gpt2' in self.model_checkpoint or 'gemma' in self.model_checkpoint:
                    is_encoder = False

                if batch['input_ids_corrupt'].shape[1] != batch['attention_mask'].shape[1]:
                    idxes.extend(batch['idx'].tolist())
                    for entry in shuffled_data.keys():
                        shuffled_data[entry].extend([None])
                    continue

                batch_lengths = batch["length"].numpy()
                if is_encoder:
                    mask_pos = (batch['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                pos4preds = mask_pos if is_encoder else torch.tensor([length - 1 for length in batch_lengths])
                shuffled_data['pos4pred'].extend(pos4preds.tolist())

                idxes.extend(batch['idx'].tolist())
                target_token = sel_dataset[batch['idx'].tolist()[0]]['target_word']
                target_tokens_pool = [target_token, target_token.capitalize(), ' ' + target_token, ' ' + target_token.capitalize()]
                target_tokens_pool = target_tokens_pool[2:] if target_token in ['him', 'himself', 'hers', 'herself'] else target_tokens_pool
                target_token_ids_pool = [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in target_tokens_pool]

                # clean run without patching
                input_batch_clean = {k: batch[k].to(DEVICE) for k in batch.keys() - ['input_ids_corrupt', 'idx', 'length', 'cues_tokenIdxes']}
                outputs_clean = model(**input_batch_clean, return_dict=False)
                prob_target_clean_pool = torch.softmax(outputs_clean[0][0, pos4preds[0], :], dim=-1)[target_token_ids_pool]
                prob_target_clean, index = torch.max(prob_target_clean_pool, dim=-1)
                target_token_id = target_token_ids_pool[index]
                predictions = outputs_clean[0]
                preds_words, preds_probs = self._extract_pred_words_probs(predictions, pos4preds)
                shuffled_data['model_top1_clean_prediction'].extend(preds_words)
                shuffled_data['model_top1_clean_confidence'].extend(preds_probs)
                shuffled_data['input_ids'].extend(batch['input_ids'])

                # corrupt run : get the corrupt value vectors for patching
                input_batch_corrupt = {k: batch[k].to(DEVICE) for k in batch.keys() - ['input_ids', 'idx', 'length', 'cues_tokenIdxes']}
                input_batch_corrupt['input_ids'] = input_batch_corrupt['input_ids_corrupt'].clone()
                del input_batch_corrupt['input_ids_corrupt']
                outputs_corrupt = model(**input_batch_corrupt, return_dict=False, output_value_vectors=True)
                predictions = outputs_corrupt[0]
                preds_words, preds_probs = self._extract_pred_words_probs(predictions, pos4preds) 
                shuffled_data['model_top1_corrupt_prediction'].extend(preds_words)
                shuffled_data['model_top1_corrupt_confidence'].extend(preds_probs)
                shuffled_data['input_ids_corrupt'].extend(batch['input_ids_corrupt'])

                # clean run with value patching : we measure how much patching each token would decrease the target token prob
                num_positions = len(batch['input_ids'][0])
                patching_result = torch.zeros((BATCH_SIZE, model.config.num_hidden_layers, num_positions))
                for layer_to_patch in range(model.config.num_hidden_layers):
                    for position_to_patch in range(num_positions):
                        outputs_clean_patched = model(**input_batch_clean,
                                                        return_dict=False,
                                                        patch_value_vector=outputs_corrupt[-1][layer_to_patch], 
                                                        patch_value_layer=layer_to_patch, 
                                                        patch_value_position=[position_to_patch, position_to_patch+1])
                        
                        prob_target_clean_patched = torch.softmax(outputs_clean_patched[0][0, pos4preds[0], :], dim=-1)[target_token_id]
                        patching_result[0, layer_to_patch, position_to_patch] = (prob_target_clean - prob_target_clean_patched).item()

                shuffled_data['vp_all_layersAndPos'].extend(patching_result)

                batch_cues_tokenIdxes = batch['cues_tokenIdxes'].numpy()
                shuffled_data["cues_tokenIdxes"].extend(batch_cues_tokenIdxes)


        # reorder retrieved data
        inverse_idxes = np.argsort(idxes)
        for key in shuffled_data.keys():
            if not shuffled_data[key]:
                shuffled_data[key] = [None for _ in range(dataset_size)]
            final_data[key] = [shuffled_data[key][inverse_idxes[i]] for i in range(dataset_size)]
            
        df = pd.DataFrame(final_data)
        return df

