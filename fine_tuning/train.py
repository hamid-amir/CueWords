import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoConfig
from datasets import load_from_disk
from tqdm import tqdm
import argparse
from modelsForPromptFinetuning import RobertaForMaskedLM, BertForMaskedLM, GPT2LMHeadModel



class FineTuner:
    MALE_PRONOUNS = ['he', 'his', 'him', 'himself']
    FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']

    def __init__(self, model_checkpoint, dataset_path, batch_size=1, epochs=2, learning_rate=5e-5):
        self.model_checkpoint = model_checkpoint
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_model_and_tokenizer()

    def _setup_model_and_tokenizer(self):
        config = AutoConfig.from_pretrained(self.model_checkpoint)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        # Add a padding token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if 'roberta' in self.model_checkpoint:
            self.model = RobertaForMaskedLM.from_pretrained(self.model_checkpoint, config=config)
        elif 'bert' in self.model_checkpoint:
            self.model = BertForMaskedLM.from_pretrained(self.model_checkpoint, config=config)
        elif 'gpt2' in self.model_checkpoint:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_checkpoint, config=config)
        else:
            raise NotImplementedError(f"{self.model_checkpoint} not implemented!")
        
        t_id = 0 if 'gpt2' in self.model_checkpoint else 1
        self.target2id = {token: self.tokenizer.encode(' ' + token)[t_id] for token in self.MALE_PRONOUNS + self.FEMALE_PRONOUNS}
        self.id2label = {token_id: i for i, token_id in enumerate(self.target2id.values())}
        self.model.label_word_list = list(self.target2id.values())
        self.model.to(self.device)

    def _suitable_mask(self, example):
        # the format is alread suitable for BERT since we used [MASK] token when constructing our dataset
        if 'roberta' in self.model_checkpoint:
            example['masked_text'] = example['masked_text'].replace('[MASK]', '<mask>')
        elif 'gpt2' in self.model_checkpoint:
            mask_start_idx = example['masked_text'].find(' [MASK]')
            if mask_start_idx == -1:
                mask_start_idx = example['masked_text'].find('[MASK]')
            example['masked_text'] = example['masked_text'][:mask_start_idx]
        return example
    
    def preprocess_function(self, examples):
        args = (examples['masked_text'],)
        result = self.tokenizer(*args, padding=False, truncation=False)
        return result

    def _check_input_length(self, example):
        config = self.model.config
        if 'gpt2' in self.model_checkpoint:
            max_input_length = config.n_positions
        else:
            max_input_length = config.max_position_embeddings 
        if len(example['input_ids']) > max_input_length:
            return False
        return True

    def prepare_data(self):
        dataset = load_from_disk(self.dataset_path)
        dataset = dataset.map(self._suitable_mask, batched=False)
        dataset = dataset.map(self.preprocess_function, batched=True, batch_size=1024)
        for split in dataset.keys():
            dataset[split] = dataset[split].add_column("target_word_id", [self.target2id[example['target_word']] for example in dataset[split]])
            dataset[split] = dataset[split].add_column("label_idx", [self.id2label[example['target_word_id']] for example in dataset[split]])
        dataset = dataset.filter(self._check_input_length, batched=False)

        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        cols = ["input_ids", "attention_mask", "label_idx"]
        dataset.set_format(type="torch", columns=cols)

        train_loader = DataLoader(
            dataset['train'],
            batch_size=self.batch_size,
            collate_fn=collator,
        )

        test_loader = DataLoader(
            dataset['test'],
            batch_size=self.batch_size,
            collate_fn=collator,
        )

        return train_loader, test_loader
    
    def train(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Fine-tuning"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label_idx'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for batch in tqdm(dataloader, desc="Evaluating on test loader"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label_idx'].to(self.device)
            
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs[0]
                pred_ids = torch.max(logits, dim=1).indices
                for pred_id, label in zip(pred_ids, labels):
                    if pred_id == label:
                        correct += 1
                    total += 1
            return f'accuracy on the test loader: {100 * correct / total}%'

    def _save_model(self):
        try:
            self.model.save_pretrained(f"./finetuned-{self.model_checkpoint.split('-')[0]}")
            self.tokenizer.save_pretrained(f"./finetuned-{self.model_checkpoint.split('-')[0]}")
            print(f'{self.model_checkpoint} fine-tuned model on the gender_agreement dataset was saved successfully!')
        except Exception as e:
            print(f'An error occurred while saving the fine-tuned model: {e}')

    def run(self):
        train_loader, test_loader = self.prepare_data()

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        print(f'{self.model_checkpoint} evaluation w/o fine-tuning and by just confining the model vocab:', self.evaluate(test_loader))
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self.train(train_loader, optimizer, scheduler)
            print(f"Training loss: {train_loss}")
            print(self.evaluate(test_loader))

        self._save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model with specified parameters.")
    parser.add_argument("model_type", choices=["roberta", "bert", "gpt2"], help="The type of model to fine-tune.")
    parser.add_argument("--dataset_path", type=str, default="data/gender_agreement", help="Path to the dataset.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")

    args = parser.parse_args()

    if args.model_type == 'roberta':
        model_ckpt = 'roberta-base'
    elif args.model_type == 'bert':
        model_ckpt = 'bert-base-uncased'
    elif args.model_type == 'gpt2':
        model_ckpt = 'gpt2'

    finetuner = FineTuner(model_ckpt, args.dataset_path, args.batch_size, args.epochs, args.learning_rate)
    finetuner.run()
