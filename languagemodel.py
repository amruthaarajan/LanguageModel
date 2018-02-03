import os
import nltk
from nltk import FreqDist, KneserNeyProbDist
from nltk.corpus import stopwords

folders = ["Pos","Neg"]
nltk.download('punkt')

#finding bigrams
def all_bigrams(my_list):
    all_bigrams = list(nltk.bigrams(my_list))
    return all_bigrams

#finding trigrams
def all_trigrams(my_list):
    all_trigrams = list(nltk.trigrams(my_list))
    return all_trigrams

#Q4) Trigram probability
def calculate_probability(trigrams):
    trigrams_as_bigrams = []
    trigrams_as_bigrams.extend([((t[0], t[1]), t[2]) for t in trigrams])

    cfd = nltk.ConditionalFreqDist(trigrams_as_bigrams)
    cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)

    # for trigram in trigrams_as_bigrams:
    #    if conditional_probability_distribution[trigram[0]].prob(trigram[1]) == 1 :
    #        print("{1} has probablity {0}".format(conditional_probability_distribution[trigram[0]].prob(trigram[1]), trigram))
    return cpd


def find_probability(cpd, trigram):
    tup = (trigram[0], trigram[1])
    value = (tup, trigram[2])
    result = cpd[value[0]].prob(value[1])
    if result == 0:
        return smoothing()
    else:
        return result

def smoothing():
    val = 1.0
    val = val / (len(trigrams) + 1)
    return val

def top_x_from_ygram(my_list,x):
    fdist = FreqDist(my_list)
    return str(fdist.most_common(x))

#filter words
def filter_words(my_list):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in my_list if not w in stop_words]
    return filtered_sentence


#Main starts here

print("\n\n\nLANGUAGE MODEL\n\n\n")

all_review_positive_words = []
all_review_negative_words = []
for folder in folders:
    path = "data/" + folder
    for filename in os.listdir(path):
        file_with_path=path + "/" + filename
        file_content = open(file_with_path).read()
        tokens = nltk.word_tokenize(file_content)
        tokens=[word.lower() for word in tokens if word.isalpha()]
        if folder == "Pos":
            all_review_positive_words = all_review_positive_words + tokens
        else:
            all_review_negative_words = all_review_negative_words + tokens


#to get filtered sentence avoiding stopwords
filtered_all_words = filter_words(all_review_positive_words) + filter_words(all_review_negative_words)


#Bigrams and Trigrams
bigrams = all_bigrams(filtered_all_words)
trigrams = all_trigrams(filtered_all_words)

#printing top 10 bigrams and trigrams
x = 10
print("Top 10 bigrams: " + top_x_from_ygram(bigrams, x))
print("Top 10 trigrams: " + top_x_from_ygram(trigrams, x))

#trigram probability of getting sequence given three words.
cpd = calculate_probability(trigrams)


#The total no of words in tokens and the number of unique words.
print("number of word tokens in the database before filtering: " + str(len(all_review_positive_words + all_review_negative_words)))
print("number of word tokens in the database after filtering: " + str(len(filtered_all_words)))
unique_words = set(filtered_all_words)
print("Vocabulary size (number of unique words) of the filtered dataset: " + str(len(unique_words)))


# Testcases
print("\n\nTestcases\n")
value1 = ('clear', 'day', 'million')
probab = find_probability(cpd, value1)
print("{1} has probablity {0}".format(probab, value1))

value2 = ('look', 'like', 'poppins')
probab = find_probability(cpd, value2)
print("{1} has probablity {0}".format(probab, value2))

value3 = ('one', 'night', 'stands')
probab = find_probability(cpd, value3)
print("{1} has probablity {0}".format(probab, value3))

value4 = ('coast', 'guard', 'rather')
probab = find_probability(cpd, value4)
print("{1} has probablity {0}".format(probab, value4))

value5 = ('next', 'two', 'years')
probab = find_probability(cpd, value5)
print("{1} has probablity {0}".format(probab, value5))

value6 = ('amruthaa', 'rajan', 'awesome')
probab = find_probability(cpd, value6)
print("{1} has probablity {0}".format(probab, value6))
