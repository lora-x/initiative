from distutils.command.build import build
import random
import numpy as np
import torch
import csv
import argparse

# for BERT
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments 
# for dialogRPT
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from datasets import load_metric
from sklearn.metrics import mean_squared_error

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
                    default=10,
                    type=int,
                    required=False,
                    help="Number of epochs to train")
    parser.add_argument("--pretrain",
                    default="dialogRPT",
                    type=str,
                    required=False,
                    help="Choose from models: BERT, dialogRPT")
    parser.add_argument("--batch_size",
                    default=4,
                    type=int,
                    required=False,
                    help="Batch size, default = 4")
    parser.add_argument("--learning_rate",
                    default=0.01,
                    type=float,
                    required=False,
                    help="learning rate, default = 0.01")


    args = parser.parse_args()

    # load from tsv into three lists
    train_bot_sents, train_human_sents, train_labels = load_tsv(args.train_dir)
    if args.dev_dir:
        dev_bot_sents, dev_human_sents, dev_labels = load_tsv(args.dev_dir)
    else: # evaluate on train set if dev set not provided
        dev_bot_sents, dev_human_sents, dev_labels = train_bot_sents, train_human_sents, train_labels

    if args.pretrain == "BERT":

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

        train_dataset = build_dataset(train_bot_sents, train_human_sents, train_labels, tokenizer)
        dev_dataset = build_dataset(dev_bot_sents, dev_human_sents, dev_labels, tokenizer)

        training_args = TrainingArguments("test_trainer", num_train_epochs=args.num_epoch, evaluation_strategy="epoch", )

        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset, compute_metrics=compute_metrics)
        trainer.train()
        # accuracy = trainer.evaluate()["eval_accuracy"]
        print (trainer.evaluate())

    elif args.pretrain == "dialogRPT":

        print("\n" * 10)

        # linear layer: {depth, width, upvote} -> {1 - 5}

        depth_model_card = "microsoft/DialogRPT-depth"
        width_model_card = "microsoft/DialogRPT-width"
        updown_model_card = "microsoft/DialogRPT-updown"

        

        depth_tokenizer = AutoTokenizer.from_pretrained(depth_model_card)
        depth_model = AutoModelForSequenceClassification.from_pretrained(depth_model_card)
        width_tokenizer = AutoTokenizer.from_pretrained(width_model_card)
        width_model = AutoModelForSequenceClassification.from_pretrained(width_model_card)
        updown_tokenizer = AutoTokenizer.from_pretrained(updown_model_card)
        updown_model = AutoModelForSequenceClassification.from_pretrained(updown_model_card)

        all_train_losses = []

        train_input_scores = [ [] ] * len(train_labels)
        for id, human_sent in enumerate(train_human_sents):
            bot_sent = train_bot_sents[id]
            depth_score = 10 * torch.logit(score_rpt(bot_sent, human_sent, depth_tokenizer, depth_model)).item()
            width_score = 10 * torch.logit(score_rpt(bot_sent, human_sent, width_tokenizer, width_model)).item()
            updown_score = 10 * torch.logit(score_rpt(bot_sent, human_sent, updown_tokenizer, updown_model)).item()
            train_input_scores[id] = [depth_score, width_score, updown_score]
            # print('depth = %.3f  width = %.3f  updown = %.3f  label = %i : %s > %s'%(depth_score, width_score, updown_score, label, bot_sent, human_sent))

        X = torch.tensor(train_input_scores) 
        Y = torch.tensor(train_labels, dtype=torch.long) # need type long for class (integer)

        rpt_net = Net(X.shape[1])
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(rpt_net.parameters(), lr=0.01)

        batch_size = args.batch_size
        for epoch in range(args.num_epoch):

            print ("\n -------------- EPOCH %i --------------" % (epoch))

            for name, param in rpt_net.named_parameters():
                if param.requires_grad:
                    print (name, " grad = ", param.grad)

            # X is a torch Variable
            permutation = torch.randperm(X.size()[0])
            epoch_loss = 0

            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:min(i+batch_size,len(permutation))]
                batch_x, batch_y = X[indices], Y[indices]

                batch_y_pred = rpt_net(batch_x)
                loss = loss_func(batch_y_pred, batch_y)
                epoch_loss += loss # record

                loss.backward()
                optimizer.step()
    
            all_train_losses.append(epoch_loss.item())
            print ("loss = ", epoch_loss.item())

        # evaluate

        # on itself 
        Y_pred = rpt_net(X)
        Y_pred_softmax = torch.log_softmax(Y_pred, dim = 1)
        _, Y_pred_tags = torch.max(Y_pred_softmax, dim = 1)
        for index, pred in enumerate(Y_pred_tags):
            print ("pred = ", pred.item(), "correct label = ", Y[index].item())
        acc, mse = multi_acc(Y_pred, Y)
        
        print ("all_train_losses = ", all_train_losses)
        print ("\n train acc = ", acc.item(), "\n train mse = ", mse.item())

        # on dev set
        # we have dev_bot_sents, dev_human_sents, dev_labels

        dev_input_scores = [ [] ] * len(dev_labels)
        for id, human_sent in enumerate(dev_human_sents):
            bot_sent = dev_bot_sents[id]
            depth_score = 10 * torch.logit(score_rpt(bot_sent, human_sent, depth_tokenizer, depth_model)).item()
            width_score = 10 * torch.logit(score_rpt(bot_sent, human_sent, width_tokenizer, width_model)).item()
            updown_score = 10 * torch.logit(score_rpt(bot_sent, human_sent, updown_tokenizer, updown_model)).item()
            dev_input_scores[id] = [depth_score, width_score, updown_score]

        dev_X = torch.tensor(dev_input_scores) 
        dev_Y = torch.tensor(dev_labels, dtype=torch.long) # need type long for class (integer)

        dev_Y_pred = rpt_net(dev_X)
        dev_Y_pred_softmax = torch.log_softmax(dev_Y_pred, dim = 1)
        _, dev_Y_pred_tags = torch.max(dev_Y_pred_softmax, dim = 1)
        for index, pred in enumerate(dev_Y_pred_tags):
            print ("pred = ", pred.item(), "correct label = ", dev_Y[index].item())
        acc, mse = multi_acc(dev_Y_pred, dev_Y)
        
        print ("\n dev acc = ", acc.item(), "\n dev mse = ", mse.item())

def multi_acc(y_pred, y):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
    correct_pred = (y_pred_tags == y).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    mse = 0
    
    return acc, mse

class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 10)
    self.fc2 = nn.Linear(10, 5)
  def forward(self, x):
    x = F.tanh(self.fc1(x)) # using tanh is better than ReLu bc lots of negative values
    return self.fc2(x) # CrossEntropyLoss expects logits (numbers running from -infinity to infinity) as its inputs, not probabilities on (0, 1)

#dialogRPT
def score_rpt(cxt, hyp, tokenizer, model):
        model_input = tokenizer.encode(cxt + "<|endoftext|>" + hyp, return_tensors="pt")
        result = model(model_input, return_dict=True)
        return torch.sigmoid(result.logits)

def load_tsv(input_file):
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        bot_sents = []
        human_sents = []
        labels = []
        
        for line in reader:
            bot_sents.append(line[0])
            human_sents.append(line[1])
            labels.append(int(line[2])-1) # convert scores from str to int & map from [1-5] to [0-4]
        return bot_sents, human_sents, labels

# BERT
class TurnDataset(Dataset):
    def __init__(self, encodings, labels=None): # setting the default labels parameter as None so that we can reuse the class to make prediction on unseen data as these data do not have labels.
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# BERT
def build_dataset(bot_sents, human_sents, labels, tokenizer):
    encodings = tokenizer(bot_sents, human_sents, padding=True, truncation=True)
    return TurnDataset(encodings, labels)

def compute_metrics(eval_pred):
    # accuracy
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    stats = metric.compute(predictions=predictions, references=labels) # return a dictionary with string keys (the name of the metric) and float values
    # MSE
    mse = mean_squared_error(labels, predictions)
    stats["MSE"] = mse
    return stats

if __name__ == "__main__":
    main()

# # see the predictions
# predictions = trainer.predict(train_dataset).predictions # output is named tuple with three fields: predictions, label_ids, and metrics
# predictions = np.argmax(predictions, axis=1)
# print ("for train")
# for pred in predictions:
#     print (pred + 1)

# predictions = trainer.predict(dev_dataset).predictions # output is named tuple with three fields: predictions, label_ids, and metrics
# predictions = np.argmax(predictions, axis=1)
# print ("for dev")
# for pred in predictions:
#     print (pred + 1)