import spacy
from gpt_sent_prob import sent_scoring, model_init

def normalize(arr):
    m = max(arr)
    for i in range(len(arr)):
        arr[i] = arr[i] / m
    
    return arr

#fin = open("processed_sentences.tsv", 'r')

def process_file(url):
    fin = open(url, 'r')
    nlp = spacy.load("en_core_web_sm")
    model, tokenizer = model_init('gpt2', False) 
    ratings = []

    all_num_tokens = []
    all_num_nouns = []
    all_nll = []
    all_ratings = []

    fin.readline()
    for l in fin.readlines():
        line = l.replace('\n',"").split('\t')
        bot = line[0]
        human = line[1]
        rating = line[2]

        num_tokens = len(human.split(" "))
        num_nouns = len([chunk.text for chunk in nlp(human).noun_chunks])
        nll = sent_scoring((model, tokenizer), human, False)
        #r = {"num_tokens": num_tokens, "num_nouns": num_nouns, "nll": nll, "human_rating": rating}
        #ratings.append(r)
        all_ratings.append(int(rating))
        all_num_tokens.append(num_tokens)
        all_num_nouns.append(num_nouns)
        all_nll.append(nll)

    all_num_tokens = normalize(all_num_tokens)
    all_num_nouns = normalize(all_num_nouns)
    all_nll = normalize(all_nll)

    return all_ratings, all_num_tokens, all_num_nouns, all_nll

# for i in range(len(all_nll)):
#     r = {"num_tokens": all_num_tokens[i], "num_nouns": all_num_nouns[i], "nll": all_nll[i], "human_rating": all_ratings[i]}
#     print(r)

# for i in range(len(all_nll)):
#     average = round((all_num_tokens[i] + all_num_nouns[i] + all_nll[i]) / 3 / 0.2)
#     average = max(average, 1)
#     if(average == all_ratings[i]):
#         accuracy += 1
#     r = {"num_tokens": all_num_tokens[i], "num_nouns": all_num_nouns[i], "nll": all_nll[i], "baseline_rating": average, "human_rating": all_ratings[i]}
#     print(r)

def train(all_ratings, all_num_tokens, all_num_nouns, all_nll):

    average_scores = []

    for i in range(len(all_nll)):
        average = (all_num_tokens[i] + all_num_nouns[i] + all_nll[i]) / 3
        average_scores.append(average)
        # r = {"num_tokens": all_num_tokens[i], "num_nouns": all_num_nouns[i], "nll": all_nll[i], "baseline_rating": average, "human_rating": all_ratings[i]}
        # print(r)

    data_length = len(all_ratings)

    best_acuracy = 0
    best_acuracy_parameter = [0, 0, 0, 0, 0]

    for l1i in range(0, 30, 1):
        for l2i in range(l1i+1, 50, 1):
            for l3i in range(l2i+1, 70, 1):
                for l4i in range(l3i+1, 90, 1):
                    l1 = l1i / 100.0
                    l2 = l2i / 100.0
                    l3 = l3i / 100.0
                    l4 = l4i / 100.0

                    acuracy = 0
                    for j in range(data_length):
                        i = average_scores[j]
                        rating = 0
                        if(i <= l1):
                            rating = 1
                        elif (i <= l2):
                            rating = 2
                        elif (i <= l3):
                            rating = 3
                        elif (i <= l4):
                            rating = 4
                        else:
                            rating = 5

                        if rating == all_ratings[j]:
                            acuracy += 1
                    
                    acuracy /= (data_length * 1.0)

                    if(acuracy > best_acuracy):
                        best_acuracy = acuracy
                        best_acuracy_parameter = [l1, l2, l3, l4]
    
    return best_acuracy, best_acuracy_parameter
                        


if __name__ == '__main__':
    all_ratings, all_num_tokens, all_num_nouns, all_nll = process_file("small.tsv")
    best_acuracy, best_acuracy_parameter = train(all_ratings, all_num_tokens, all_num_nouns, all_nll)
    print(best_acuracy)
    print(best_acuracy_parameter)