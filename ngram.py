import os
import nltk
from nltk.util import ngrams

folders = ["Pos","Neg"]
nltk.download('punkt')

def top_bigrams(my_list):
    print("In bigram.. ")
    return

def top_trigrams(my_list):
    print("In trigram.. ")
    return


all_review_positive_words = []
all_review_negative_words = []
for folder in folders:
    path = "data/" + folder
    print("In folder " + folder + "...")
    for filename in os.listdir(path):
        file_with_path=path + "/" + filename
        #print(file_with_path)
        file_content = open(file_with_path).read()
        tokens = nltk.word_tokenize(file_content)
        if folder == "Pos":
            all_review_positive_words = all_review_positive_words + tokens
        else:
            all_review_negative_words = all_review_negative_words + tokens
    if (len(all_review_positive_words) > 0) and (len(all_review_negative_words) == 0):
        top_bigrams(all_review_positive_words)
        top_trigrams(all_review_positive_words)
    else:
        top_bigrams(all_review_negative_words)
        top_trigrams(all_review_negative_words)

all_words = all_review_positive_words + all_review_negative_words
print("number of word tokens in the database: " + str(len(all_words)))
unique_words = set(all_words)
print("Vocabulary size (number of unique words) of the dataset: " + str(len(unique_words)))
