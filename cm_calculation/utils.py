# put the following dir into the current dir
# !git clone https://github.com/hmohebbi/context_mixing_toolkit.git -q


from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import random
from IPython.display import display
import numpy as np
import torch
from typing import List, Tuple
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from context_mixing_toolkit.src.modeling_bert import BertModel
from context_mixing_toolkit.src.modeling_roberta import RobertaModel
from context_mixing_toolkit.src.modeling_gemma import GemmaModel
from context_mixing_toolkit.src.utils import CMConfig, normalize, rollout



class dataset2cuesCM:
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

            word_token_positions = []
            for i, (start, end) in enumerate(offsets):
                if cue_start <= start <= cue_end and cue_start <= end <= cue_end and start < end:
                    word_token_positions.append(i)
            cues_tokenIdxes.append([word_token_positions[0], word_token_positions[-1]+1])

        return  cues_tokenIdxes
    

    def _suitable_mask(self, examples):
        if 'roberta' in self.model_checkpoint:
            examples['masked_text'] = [text.replace('[MASK]', '<mask>') for text in examples['masked_text']]
        return examples


    def _preprocess_function_wrapped(self, tokenizer):
        def preprocess_function(examples):
            # Tokenize the texts
            args = (examples['masked_text'],)
            result = tokenizer(*args, padding=False)
            return result
        return preprocess_function


    def _check_input_length_wrapped(self, model):
        def check_input_length(example):
            config = model.config
            max_input_length = config.max_position_embeddings - 2
            if len(example['input_ids']) > max_input_length:
                return False
            return True
        return check_input_length


    def _extract_cue_words_cm(self, cm, batch_lengths, batch_cues_tokenIdxes):
        cue_words_cm = []
        for c in range(self.num_cues):
            cue_tokens_cm = [cm[j][:, batch_lengths[j]-2, batch_cues_tokenIdxes[j][c][0]: batch_cues_tokenIdxes[j][c][1]] for j in range(len(cm))]
            cue_word_cm = np.array([np.max(cue_token_cm, axis=1) for cue_token_cm in cue_tokens_cm])
            cue_words_cm.append(torch.tensor(cue_word_cm))
        return cue_words_cm


    def retrieveCM(self) -> pd.DataFrame:
        """
        Main function that retrievs context mixing scores of cue words based on
        various interpretability methods.
        """
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        BATCH_SIZE = 16

        # load gender_agreement dataset
        dataset = load_dataset(self.dataset_path.split('.')[-1], data_files=self.dataset_path)

        # seperate examples that has the given number of cue words
        sel_dataset = dataset['train'].filter(lambda example: len(example['cue_words'])==self.num_cues)

        # set up the model
        if "roberta" in self.model_checkpoint:
            model = RobertaModel.from_pretrained(self.model_checkpoint)
        elif "bert" in self.model_checkpoint:
            model = BertModel.from_pretrained(self.model_checkpoint)
        else:
            print(self.model_checkpoint + ' not implemented!')

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)

        # add index of the each example in the dataset 
        sel_dataset = sel_dataset.add_column("idx", [i for i in range(len(sel_dataset))])

        # each model has it's own mask token
        sel_dataset = sel_dataset.map(self._suitable_mask, batched=True, batch_size=1024)

        # add tokenization output of each sample to the dataset
        sel_dataset = sel_dataset.map(self._preprocess_function_wrapped(tokenizer), batched=True, batch_size=1024)

        # ensure that each example is shorter than the model max input length
        sel_dataset = sel_dataset.filter(self._check_input_length_wrapped(model), batched=False)

        dataset_size = len(sel_dataset)
        steps = int(np.ceil(dataset_size / BATCH_SIZE))

        # we will output this final_data as a pandas dataframe
        final_data = {
            "masked_text": [], # str
            "target_word": [], # str
            "cue_words": []    # List[str]
        }

        lengths = []
        all_cues_tokenIdxes = []

        for i in tqdm(range(dataset_size), desc="Initializing final data and extracting cue words tokens indices"):

            masked_text = sel_dataset[i]['masked_text']
            target_word = sel_dataset[i]['target_word']
            cue_words = sel_dataset[i]['cue_words']

            final_data["masked_text"].append(masked_text)
            final_data["target_word"].append(target_word)
            final_data["cue_words"].append(cue_words)


            length = len(sel_dataset[i]['input_ids'])
            cues_tokenIdxes = self._get_token_idxes_for_cues(tokenizer, sel_dataset[i])

            lengths.append(length)
            all_cues_tokenIdxes.append(cues_tokenIdxes)


        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        sel_dataset = sel_dataset.add_column("length", lengths)
        sel_dataset = sel_dataset.add_column("cues_tokenIdxes", all_cues_tokenIdxes)
        cols = ["input_ids", "attention_mask", "idx", "length", "cues_tokenIdxes"]
        cols = cols + ["token_type_ids"] if not "roberta" in self.model_checkpoint else cols
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

        # we are going to retrieve these values
        shuffled_data = {
            "cueWords_attention_CM_all_layers": [],      # tensor[layer, num_cues]
            "cueWords_rollout_CM_all_layers": [],        # tensor[layer, num_cues]
            "cueWords_attentionNorm_CM_all_layers": [],  # tensor[layer, num_cues]
        }

        with torch.no_grad():
            for i in tqdm(range(steps), desc="Forwarding and extracting cue words context mixing scores"):
                batch = next(it)
                input_batch = {k: batch[k].to(DEVICE) for k in batch.keys() - ['idx', 'length', 'cues_tokenIdxes']}
                cm_config = CMConfig(output_attention=True, output_attention_norm=True)
                outputs = model(**input_batch, output_context_mixings=cm_config)


                idxes.extend(batch['idx'].tolist())
                batch_lengths = batch["length"].numpy()
                batch_cues_tokenIdxes = batch['cues_tokenIdxes'].numpy()

                #  these cm have shape => (batch_size, layer, max_seqLen_batch, max_seqLen_batch)
                attention_cm = torch.stack(outputs['context_mixings']['attention']).permute(1, 0, 2, 3).detach().cpu().numpy()
                rollout_cm = np.array([rollout(attention_cm[j], res=True) for j in range(len(attention_cm))])
                attentionNorm_cm = normalize(torch.stack(outputs['context_mixings']['attention_norm']).permute(1, 0, 2, 3).detach().cpu().numpy())

                cueWords_attention_cm = self._extract_cue_words_cm(attention_cm, batch_lengths, batch_cues_tokenIdxes)
                cueWords_rollout_cm = self._extract_cue_words_cm(rollout_cm, batch_lengths, batch_cues_tokenIdxes)
                cueWords_attentionNorm_cm = self._extract_cue_words_cm(attentionNorm_cm, batch_lengths, batch_cues_tokenIdxes)


                # these cue_words_cm have shape => (batch_size, layer, num_cues)
                shuffled_data['cueWords_attention_CM_all_layers'].extend(torch.stack(cueWords_attention_cm, dim=2))
                shuffled_data['cueWords_rollout_CM_all_layers'].extend(torch.stack(cueWords_rollout_cm, dim=2))
                shuffled_data['cueWords_attentionNorm_CM_all_layers'].extend(torch.stack(cueWords_attentionNorm_cm, dim=2))

        # reorder retrieved data
        inverse_idxes = np.argsort(idxes)
        for key in shuffled_data.keys():
            if len(shuffled_data[key]) == 0:
                shuffled_data[key] = [None for _ in range(dataset_size)]
            final_data[key] = [shuffled_data[key][inverse_idxes[i]] for i in range(dataset_size)]
            
        df = pd.DataFrame(final_data)
        return df
    