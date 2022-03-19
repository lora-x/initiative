import csv
from datasets import load_metric
from sklearn.metrics import mean_squared_error
import numpy as np
import torch

def load_tsv(input_file, method="BERT"):
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")

        if method == "BERT":
            bot_sents = []
            human_sents = []
            labels = []
            
            for line in reader:
                if (int(line[2]) != 6 and int(line[2]) != 7):
                    bot_sents.append(line[0])
                    human_sents.append(line[1])
                    labels.append(int(line[2])-1) # convert scores from str to int & map from [1-5] to [0-4]
            return bot_sents, human_sents, labels

        if method == "GPT":
            turns = []
            labels = []
            turns6 = []
            labels6 = []
            turns7 = []
            labels7 = []

            prev_human = "_"
            for line in reader:
                score = int(line[2])

                turn6 = prev_human + "<|endoftext|>" + line[0]
                label6 = 1 if score == 6 else 0
                prev_human = line[1]
                turns6.append(turn6)
                labels6.append(label6)
                
                turn7 = line[0] + "<|endoftext|>" + line[1]
                label7 = 1 if score == 7 else 0
                turns7.append(turn7)
                labels7.append(label7)

                if(score != 6 and score != 7):
                    turn = line[0] + "<|endoftext|>" + line[1]
                    turns.append(turn)
                    labels.append(score-1)

            return turns, labels, turns6, labels6, turns7, labels7         

def compute_softmax_metrics(eval_pred):
    # accuracy
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    stats = metric.compute(predictions=predictions, references=labels) # return a dictionary with string keys (the name of the metric) and float values
    # MSE
    mse = mean_squared_error(labels, predictions)
    stats["MSE"] = mse
    return stats


def compute_rounded_logits_metrics(eval_pred):
    # accuracy
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    logits = torch.tensor(logits, dtype=float)
    predictions = torch.clamp(torch.round(logits), min = 0, max = 4)
    stats = metric.compute(predictions=predictions, references=labels) # return a dictionary with string keys (the name of the metric) and float values
    # MSE
    mse = mean_squared_error(labels, predictions)
    stats["MSE"] = mse
    return stats