import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
from transformers import AutoConfig
from datasets import load_dataset
from tqdm import tqdm
from modelsForPromptFinetuning import RobertaForMaskedLM, BertForMaskedLM





def train(model, dataloader, optimizer, scheduler, device):
    model = model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Fine-tuning"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label_idx'].to(device)

        outputs = model(
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


def evaluate(model, dataloader, device):
    model = model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for batch in tqdm(dataloader, desc="Evaluating on test loader"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_idx'].to(device)
        
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs[0]
            pred_ids = torch.max(logits, dim=1).indices
            for pred_id, label in zip(pred_ids, labels):
                if pred_id == label:
                    correct += 1
                total += 1
        return f'accuracy on the test loader: {100 * correct/total}%'


def _suitable_mask(examples):
    examples['masked_text'] = [text.replace('[MASK]', '<mask>') for text in examples['masked_text']]
    return examples

def _preprocess_function_wrapped(tokenizer):
    def preprocess_function(examples):
        # Tokenize the texts
        args = (examples['masked_text'],)
        result = tokenizer(*args, padding=False, truncation=False)
        return result
    return preprocess_function

def _check_input_length_wrapped(model):
    def check_input_length(example):
        config = model.config
        max_input_length = config.max_position_embeddings - 2
        if len(example['input_ids']) > max_input_length:
            return False
        return True
    return check_input_length


MALE_PRONOUNS = ['he', 'his', 'him', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']


def main(model_checkpoint):
    BATCH_SIZE = 8
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create config
    config = AutoConfig.from_pretrained(model_checkpoint)

    # Initialize the model and tokenizer
    if 'roberta' in model_checkpoint:
        tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
        model = RobertaForMaskedLM.from_pretrained(model_checkpoint, config=config)
    elif 'bert' in model_checkpoint:
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
        model = BertForMaskedLM.from_pretrained(model_checkpoint, config=config) 
    else:
        print(model_checkpoint + ' not implemented!')      

    target2id = {token:tokenizer.encode(' '+token)[1] for token in MALE_PRONOUNS + FEMALE_PRONOUNS}
    id2label = {token_id: i for i,token_id in enumerate(target2id.values())}
    model.label_word_list = list(target2id.values())
    model.to(device)

    # Prepare data
    dataset = load_dataset('json', data_files='data/'+'gender_agreement.json', split='train')
    dataset = dataset.map(_suitable_mask, batched=True, batch_size=1024)
    dataset = dataset.map(_preprocess_function_wrapped(tokenizer), batched=True, batch_size=1024)
    # dataset = dataset.add_column("mask_pos", [len(example['input_ids'])-2 for example in dataset])
    dataset = dataset.add_column("target_word_id", [target2id[example['target_word']] for example in dataset])
    dataset = dataset.add_column("label_idx", [id2label[example['target_word_id']] for example in dataset])
    dataset = dataset.filter(_check_input_length_wrapped(model), batched=False)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    cols = ["input_ids", "attention_mask", "label_idx"]
    dataset.set_format(type="torch", columns=cols)

    train_loader = DataLoader(
        dataset['train'],
        batch_size=BATCH_SIZE,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        dataset['test'],
        batch_size=BATCH_SIZE,
        collate_fn=collator,
    )

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 2 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    print(f'{model_checkpoint} evaluation w/o fine-tuning:', evaluate(model, test_loader, device))
    epochs = 2
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss}")
        print(evaluate(model, test_loader, device))

    # Save the model
    try:
        model.save_pretrained(f"./finetuned_{model_checkpoint.split('-')[0]}")
        tokenizer.save_pretrained(f"./finetuned_{model_checkpoint.split('-')[0]}")
        print(f'{model_checkpoint} fine-tuned on the gender_agreement dataset was saved successfully!')
    except:
        print('An error occured while saving the fine-tuned model.')



if __name__ == "__main__":
    main('roberta-base')
    # main('bert-base-uncased')
