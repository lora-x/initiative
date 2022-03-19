# for dialogRPT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Trainer, TrainingArguments 
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import Dataset
import wandb



from utils import load_tsv, compute_rounded_logits_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The train data dir. in .tsv")
    parser.add_argument("--dev_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The dev data dir. in .tsv")
    parser.add_argument("--test_dir",
                    default=None,
                    type=str,
                    required=False,
                    help="The test data dir. in .tsv")
    parser.add_argument("--num_epoch",
                    default=1,
                    type=int,
                    required=False,
                    help="Number of epochs to train")
    parser.add_argument("--pretrain",
                    default="depth",
                    type=str,
                    required=False,
                    help="Choose from models: depth, width, updown")
    parser.add_argument("--batch_size",
                    default=4,
                    type=int,
                    required=False,
                    help="Batch size, default = 4")
    parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    required=False,
                    help="learning rate, default = 5e-5")
    parser.add_argument("--freeze_decoder",
                    default=False,
                    type=bool,
                    required=False,
                    help="whether to freeze the decoder params and only finetune the top linear level, default = False")
    args = parser.parse_args()


    train_turns, train_labels, train_turns6, train_labels6, train_turns7, train_labels7 = load_tsv(args.train_dir, method="GPT")
    if args.dev_dir:
        print ("Loading seperate dev set...")
        dev_turns, dev_labels, dev_turns6, dev_labels6, dev_turns7, dev_labels7 = load_tsv(args.dev_dir, method="GPT")
    else: # evaluate on train set if dev set not provided
        dev_turns, dev_labels, dev_turns6, dev_labels6, dev_turns7, dev_labels7 = train_turns, train_labels, train_turns6, train_labels6, train_turns7, train_labels7

    train_all_predictions = dict()
    dev_all_predictions = dict()

    dialogRPT_models = {"depth":"microsoft/DialogRPT-depth", "width":"microsoft/DialogRPT-width", "updown":"microsoft/DialogRPT-updown"}
    model_card = dialogRPT_models[args.pretrain]
    wandb.init(project=args.pretrain, entity="lora")

    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model = AutoModelForSequenceClassification.from_pretrained(model_card)

    # freeze decoder layers if so choose
    if args.freeze_decoder:
        for name, param in model.named_parameters():
            if not (name.startswith("transformer.ln") or name.startswith("score")): 
                param.requires_grad = False

    train_dataset = build_dataset(train_turns, train_labels, tokenizer)
    dev_dataset = build_dataset(dev_turns, dev_labels, tokenizer)

    training_args = TrainingArguments(args.pretrain + "output", num_train_epochs=args.num_epoch, evaluation_strategy="epoch", learning_rate = args.learning_rate, report_to="wandb")
    trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset, compute_metrics=compute_rounded_logits_metrics)
    trainer.train()
    # accuracy = trainer.evaluate()["eval_accuracy"]
    stats = trainer.evaluate()
    print (stats)

    # emsemble
    train_prediction_output = trainer.predict(test_dataset = train_dataset)
    train_logits = torch.tensor(train_prediction_output.predictions)
    train_pred = torch.clamp(torch.round(train_logits), min = 0, max = 4)
    print ("\n \n ------------------------ train predictions ------------------------")
    for index, line in enumerate(train_turns):
        print (line, "   label = ", round(train_labels[index]), "   pred = ", train_pred[index].item())
    
    dev_prediction_output = trainer.predict(test_dataset = dev_dataset)
    dev_logits = torch.tensor(dev_prediction_output.predictions)
    dev_pred = torch.clamp(torch.round(dev_logits), min = 0, max = 4)
    print ("\n \n ------------------------ dev predictions ------------------------")
    for index, line in enumerate(dev_turns):
            print (line, "   label = ", round(dev_labels[index]), "   pred = ", dev_pred[index].item())

def build_dataset(turns, labels, tokenizer):
    encodings = tokenizer(turns, padding=True, truncation=True, return_tensors="pt")
    return TurnDataset(encodings, labels)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        predictions = torch.clamp(torch.round(logits), min = 0, max = 4)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class TurnDataset(Dataset):
    def __init__(self, encodings, labels=None): # setting the default labels parameter as None so that we can reuse the class to make prediction on unseen data as these data do not have labels.
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor([self.labels[idx]], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


# features = depth_tokenizer("This is a test")
# outputs = depth_model(**features)
# print(outputs)


#feature_extractor = pipeline("feature-extraction", model="microsoft/DialogRPT-depth", tokenizer="microsoft/DialogRPT-depth")
# features = feature_extractor(["I love NLP!<|endoftext|>Me too!", "I love NLP!<|endoftext|>Here’s a free textbook (URL) in case anyone needs it."])
# features = feature_extractor("I love NLP!<|endoftext|>Me too!")
# features = feature_extractor("I love NLP!<|endoftext|>Here’s a free textbook (URL) in case anyone needs it.")
#features = feature_extractor("Hi<|endoftext|>Hello")
#features = np.squeeze(np.array(features))
#features = np.around(features)

# print (features[0])
#print(np.shape(features))
#print(features)

if __name__ == "__main__":
    main()