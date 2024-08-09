# put the following dir into the current dir
# !git clone https://github.com/hmohebbi/context_mixing_toolkit.git -q


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
from transformers import AutoTokenizer
from context_mixing_toolkit.src.modeling_bert import BertModel
from context_mixing_toolkit.src.modeling_roberta import RobertaModel
from context_mixing_toolkit.src.modeling_gpt2 import GPT2Model
from context_mixing_toolkit.src.modeling_gemma import GemmaModel
from context_mixing_toolkit.src.utils import CMConfig, normalize, rollout
from transformers import RobertaForMaskedLM, BertForMaskedLM, GemmaForCausalLM, GPT2LMHeadModel



class dataset2CMs:
    def __init__(
            self,
            model_checkpoint: str,
            dataset_path: str,
            num_cues: int
    ) -> pd.DataFrame:
        self.model_checkpoint = model_checkpoint
        self.dataset_path = dataset_path
        self.num_cues = num_cues

    
    def _get_token_idxes_for_cues(self, tokenizer, sample) -> List[List[Tuple[int, int]]]:
        """
        Returns start and end indices of tokens corresponding to each cue word.
        Note that indices are based on input_ids list.
        """
        text = sample['masked_text']
        encoded = tokenizer.encode_plus(text, return_offsets_mapping=True)
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
    

    def _suitable_mask(self, example):
        # the format is alread suitable for BERT since we used [MASK] token when constructing our dataset
        if 'roberta' in self.model_checkpoint:
            example['masked_text'] = example['masked_text'].replace('[MASK]', '<mask>')
        elif 'gemma' in self.model_checkpoint or 'gpt2' in self.model_checkpoint:
            mask_start_idx = example['masked_text'].find(' [MASK]')
            if mask_start_idx == -1:
                mask_start_idx = example['masked_text'].find('[MASK]')
            example['masked_text'] = example['masked_text'][:mask_start_idx]
            # examples['masked_text'] = [text.replace(' [MASK]', '') for text in examples['masked_text']]
            # examples['masked_text'] = [text.replace('[MASK]', '') for text in examples['masked_text']]
        return example


    def _preprocess_function_wrapped(self, tokenizer):
        def preprocess_function(examples):
            # Tokenize the texts
            args = (examples['masked_text'],)
            result = tokenizer(*args, padding=False, truncation=False)
            return result
        return preprocess_function


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
    

    def _extract_pred_words_probs(self, predictions, pos4preds, tokenizer):
        preds_idxes = [torch.topk(predictions[j, pos4preds[j]], 1).indices.tolist()[0] for j in range(len(predictions))]
        preds_words = [tokenizer.decode([idx]) for idx in preds_idxes]
        preds_probs = [torch.softmax(predictions, dim=0)[j, pos4preds[j], idx].item() for j, idx in enumerate(preds_idxes)]
        
        return preds_words, preds_probs



    def retrieveCM(self) -> pd.DataFrame:
        """
        Main function that retrievs context mixing scores of cue words based on
        various interpretability methods.
        """
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 1
        # if self.num_cues > 8:
        #     BATCH_SIZE = 1

        # set up the model
        if "roberta" in self.model_checkpoint:
            # first load the model with masked lm head and then save it's head
            model = RobertaForMaskedLM.from_pretrained(self.model_checkpoint)
            head = model.lm_head
            del model
            model = RobertaModel.from_pretrained(self.model_checkpoint)
        elif "bert" in self.model_checkpoint:
            model = BertForMaskedLM.from_pretrained(self.model_checkpoint)
            head = model.cls
            del model
            model = BertModel.from_pretrained(self.model_checkpoint)
        elif "gpt2" in self.model_checkpoint:
            model = GPT2LMHeadModel.from_pretrained(self.model_checkpoint)
            head = model.lm_head
            del model
            model = GPT2Model.from_pretrained(self.model_checkpoint)
        elif "gemma" in self.model_checkpoint:
            model = GemmaForCausalLM.from_pretrained(self.model_checkpoint, attn_implementation='eager')
            head = model.lm_head
            del model
            model = GemmaModel.from_pretrained(self.model_checkpoint, attn_implementation='eager')
        else:
            print(self.model_checkpoint + ' not implemented!')

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        # Add a padding token if it's not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load test split of our gender_agreement dataset
        dataset = load_from_disk(self.dataset_path)['test']

        # seperate examples that has the given number of cue words
        sel_dataset = dataset.filter(lambda example: len(example['cue_words'])==self.num_cues)

        # make the dataset size equals to the size of the dataset with 6 cues -> making the dataset balanced
        sel_dataset = sel_dataset.select(range(sel_dataset[0]['balance_size']))

        # each model has it's own mask token
        sel_dataset = sel_dataset.map(self._suitable_mask, batched=False)

        # add tokenization output of each sample to the dataset
        sel_dataset = sel_dataset.map(self._preprocess_function_wrapped(tokenizer), batched=True, batch_size=1024)

        # ensure that each example is shorter than the model max input length
        sel_dataset = sel_dataset.filter(self._check_input_length_wrapped(model), batched=False)

        all_cues_tokenIdxes = []
        for i in tqdm(range(len(sel_dataset)), desc="Extracting cue words tokens indices"):
            cues_tokenIdxes = self._get_token_idxes_for_cues(tokenizer, sel_dataset[i])
            all_cues_tokenIdxes.append(cues_tokenIdxes)

        sel_dataset = sel_dataset.add_column("cues_tokenIdxes", all_cues_tokenIdxes)

        # added to make the comparison between cm and vp results fair
        sel_dataset = sel_dataset.filter(self._excludeForCorrupt, batched=False)

        # add index of the each example in the dataset 
        sel_dataset = sel_dataset.add_column("idx", [i for i in range(len(sel_dataset))])

        # we will output this final_data as a pandas dataframe
        final_data = {
            "masked_text": [], # List[str]
            "target_word": [], # List[str]
            "cue_words": []    # List[List[str]]
        }

        lengths = []
        for i in tqdm(range(len(sel_dataset)), desc="Initializing final data"):

            final_data["masked_text"].append(sel_dataset[i]['masked_text'])
            final_data["target_word"].append(sel_dataset[i]['target_word'])
            final_data["cue_words"].append(sel_dataset[i]['cue_words'])

            length = len(sel_dataset[i]['input_ids'])
            lengths.append(length)


        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        sel_dataset = sel_dataset.add_column("length", lengths)

        dataset_size = len(sel_dataset)
        steps = int(np.ceil(dataset_size / BATCH_SIZE))

        cols = ["input_ids", "attention_mask", "idx", "length", "cues_tokenIdxes"]
        sel_dataset.set_format(type="torch", columns=cols)


        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        sampler = LengthGroupedSampler(
            BATCH_SIZE,
            lengths=lengths,
            model_input_name=tokenizer.model_input_names[0],
            generator=generator,
        )


        dataloader = DataLoader(
            sel_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            collate_fn=collator,
        )


        model.to(DEVICE)
        it = iter(dataloader)
        idxes = []

        # we are going to retrieve these values for the target token of each example
        shuffled_data = {
            "input_ids": [], # List[tensor(seq_len i.e len of input_ids w/o any padding)]
            "pos4pred": [],  # List[int]

            "model_top1_prediction": [],  # List[str]
            "model_top1_confidence": [],  # List[float]

            "attention_cm_all_layers": [],   # List[tensor(layer, seq_len i.e len of input_ids w/o any padding)]
            "rollout_cm_all_layers": [],        
            "attentionNorm_cm_all_layers": [],  
            "attentionNormRes1_cm_all_layers": [],
            "attentionNormRes1Ln1_cm_all_layers": [],
            "valueZeroing_cm_all_layers": [],

            "cues_tokenIdxes": [] # List[array(num_cues, 2)]
        }

        with torch.no_grad():
            for i in tqdm(range(steps), desc="Forwarding and extracting cue words context mixing scores"):
                batch = next(it)
                input_batch = {k: batch[k].to(DEVICE) for k in batch.keys() - ['idx', 'length', 'cues_tokenIdxes']}
                if 'roberta' in self.model_checkpoint or 'bert' in self.model_checkpoint:
                    is_encoder = True
                    cm_config = CMConfig(output_attention=True, output_attention_norm=True, output_value_zeroing=True)
                elif 'gpt2' in self.model_checkpoint or 'gemma' in self.model_checkpoint:
                    is_encoder = False
                    cm_config = CMConfig(output_attention=True, output_value_zeroing=True)
                outputs = model(**input_batch, output_context_mixings=cm_config)

                shuffled_data['input_ids'].extend(batch['input_ids'])

                idxes.extend(batch['idx'].tolist())
                batch_lengths = batch["length"].numpy()
                batch_cues_tokenIdxes = batch['cues_tokenIdxes'].numpy()

                # predictions shape => (batch_size, max_seqLen_batch, vocab_size)
                head = head.to(DEVICE)
                predictions = head(outputs['last_hidden_state'])
                if is_encoder:
                    mask_pos = (batch['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                pos4preds = mask_pos if is_encoder else torch.tensor([length - 1 for length in batch_lengths])
                shuffled_data['pos4pred'].extend(pos4preds.tolist())
                preds_words, preds_probs = self._extract_pred_words_probs(predictions, pos4preds, tokenizer)
                shuffled_data['model_top1_prediction'].extend(preds_words)
                shuffled_data['model_top1_confidence'].extend(preds_probs)

                #  these cm have shape => (batch_size, layer, max_seqLen_batch, max_seqLen_batch)
                cms = {}
                cms['attention_cm'] = torch.stack(outputs['context_mixings']['attention']).permute(1, 0, 2, 3).detach().cpu().numpy()
                cms['rollout_cm'] = np.array([rollout(cms['attention_cm'][j], res=True) for j in range(len(cms['attention_cm']))])
                if is_encoder:
                    cms['attentionNorm_cm'] = normalize(torch.stack(outputs['context_mixings']['attention_norm']).permute(1, 0, 2, 3).detach().cpu().numpy())
                    cms['attentionNormRes1_cm'] = normalize(torch.stack(outputs['context_mixings']['attention_norm_res']).permute(1, 0, 2, 3).detach().cpu().numpy())
                    cms['attentionNormRes1Ln1_cm'] = normalize(torch.stack(outputs['context_mixings']['attention_norm_res_ln']).permute(1, 0, 2, 3).detach().cpu().numpy())
                cms['valueZeroing_cm'] = normalize(torch.stack(outputs['context_mixings']['value_zeroing']).permute(1, 0, 2, 3).detach().cpu().numpy())

                for cm in cms.keys():
                    for j in range(len(cms[cm])):
                        shuffled_data[f"{cm}_all_layers"].append(torch.tensor(cms[cm][j, : , pos4preds[j] , :batch_lengths[j]]))
                shuffled_data["cues_tokenIdxes"].extend(batch_cues_tokenIdxes)

        # reorder retrieved data
        inverse_idxes = np.argsort(idxes)
        for key in shuffled_data.keys():
            if len(shuffled_data[key]) == 0:
                shuffled_data[key] = [None for _ in range(dataset_size)]
            final_data[key] = [shuffled_data[key][inverse_idxes[i]] for i in range(dataset_size)]
            
        df = pd.DataFrame(final_data)
        return df
    