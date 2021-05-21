#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 01:24:18 2020

@author: russell
"""

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from SupervisedAlgorithm import Logistic_Regression_Classifier
from SupervisedAlgorithm import SGD_Classifier
from SupervisedAlgorithm  import TF_IDF
from SupervisedAlgorithm import  Performance
import random
import os 


review_directory = "/Users/russell/Documents/NLP/DownloadedDataFromYoutube/Nov_14/"

#nlp_english = spacy.load('en')

vader_lexicon = {}
bing_liu_lexicon = {}
sentic_net = {}



def split_bengali_english_review(review):
    bangla_review = []
    english_review = []
    
    for text in review:
        if (len(text) < 5):
            continue
        text= text.split("~")
        #print(text)
        bangla =  text[0]
        english =  text[1]
       # print(bangla, english)
        bangla_review.append(bangla)
        english_review.append(english)
    
    return bangla_review, english_review



def get_reviews(directory_path= review_directory, filename=None):
    
    reviews = []
  
    if ".DS_Store" in filename:
        return reviews
    filename = directory_path + filename
    with open(filename) as text:
        for line in text:
           # line = line.replace("\n","")
            #print(line)
            if (len(line) < 5):
                continue
            reviews.append(line)
    
    #sprint(len(reviews))
    #reviews.append("############\n")
    return reviews


#https://www.researchgate.net/publication/338924123_Bangla_Dataset_for_OpinionMining
def get_reviews_social_media_news_paper():
    
    reviews = []
  
    filename = "/Users/russell/Downloads/BanglaDatasetOpinionMining.txt" 
    #directory_path + filename
    
    negative_reviews = []
    positive_reviews = []
    
    with open(filename) as text:
        for line in text:
           # line = line.replace("\n","")
            #print(line)
            if (len(line) < 5):
                continue
            tokens = line.split()
            
            sentiment = tokens[0]
            #print(sentiment)
            
            if sentiment == 'pos':
                positive_reviews.append(line[4:])
            
            if sentiment == 'neg':
                negative_reviews.append(line[4:])
                
            
            #reviews.append(line)
            
            
    file = open("/Users/russell/Downloads/neg_social_new.txt", 'w') 
    for review in negative_reviews: 
       file.write(review)
       
    file = open("/Users/russell/Downloads/pos_social_new.txt", 'w') 
    for review in positive_reviews: 
       file.write(review)
    
    print(len(positive_reviews), "^^^",len(negative_reviews))
    #reviews.append("############\n")
    return positive_reviews, negative_reviews


def get_mendely_paper():
    
    reviews = []
  
    filename = "/Users/russell/Documents/NLP/Paper-4/Lexicons/Mendely_News/finaldataset.txt" 
    #directory_path + filename
    
    negative_reviews = []
    positive_reviews = []
    
    with open(filename) as text:
        for line in text:
           # line = line.replace("\n","")
            #print(line)
            if (len(line) < 5):
                continue
            tokens = line.split(",")
            
            #sentiment = tokens[0]
            review = tokens[0]
            
            sentiment = tokens[-1]
            print(review)
            
            if 'নিশ্চিত নেতিবাচক' in sentiment:
                negative_reviews.append(review)
            
            if 'নিশ্চিত ইতিবাচক' in sentiment:
                positive_reviews.append(review)
                
            
            #reviews.append(line)
            
    
    file = open("/Users/russell/Downloads/neg_mendely.txt", 'w') 
    for review in negative_reviews: 
       file.write(review + "\n")
       
    file = open("/Users/russell/Downloads/pos_mendely.txt", 'w') 
    for review in positive_reviews: 
       file.write(review + "\n")
 
    
    print(len(positive_reviews), "^^^",len(negative_reviews))
    #reviews.append("############\n")
    return positive_reviews, negative_reviews

def get_review_data_from_directory(negative_directory, positive_directory):

    negative_review = []
    positive_review = []
    for filename in os.listdir(negative_directory):
        if "negative" in filename: 
            #print("N: ",filename)
            negative_review = negative_review + get_reviews(negative_directory,filename)
        
    for filename in os.listdir(positive_directory):
        if "positive" in filename:
            #print("P: ",filename)
            positive_review = positive_review + get_reviews(positive_directory,filename)
        
    return negative_review, positive_review
    
def read_final_reviews_data_between_0_75():
    positive_review = []
    negative_review = []
    labels = []
    

    negative_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/bins_w_label_correction/0.0_0.25/"
    positive_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/bins_w_label_correction/0.0_0.25/"
        
    negative, positive = get_review_data_from_directory(negative_directory, positive_directory)
    negative_review = negative_review + negative
    positive_review = positive_review + positive
     
    print("\n 1 Positive: ",len(positive_review))
    print(" 1 Negative: ",len(negative_review))
    
    
    negative_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/bins_w_label_correction/0.5_0.75/"
    positive_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/bins_w_label_correction/0.5_0.75/"
      
    negative, positive = get_review_data_from_directory(negative_directory, positive_directory)
     
    negative_review = negative_review + negative
    positive_review = positive_review + positive
     
    print("\n 2 Positive: ",len(positive_review))
    print(" 2 Negative: ",len(negative_review))
    
            
    negative_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/bins_w_label_correction/0.25_0.5/"
    positive_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/bins_w_label_correction/0.25_0.5/"
      
    negative, positive = get_review_data_from_directory(negative_directory, positive_directory)
     
    negative_review = negative_review + negative
    positive_review = positive_review + positive
     
    print("\n 3 Positive: ",len(positive_review))
    print(" 3 Negative: ",len(negative_review))
    
    
    
    negative_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/0.0/0_updated_correct_label/"
    positive_directory = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/raw/0.0/0_updated_correct_label/"
    
    negative, positive = get_review_data_from_directory(negative_directory, positive_directory)
     
    negative_review = negative_review + negative
    positive_review = positive_review + positive
     
    print("\n 4 Positive: ",len(positive_review))
    print(" 4 Negative: ",len(negative_review))
    
    bangla_neg, english = split_bengali_english_review(negative_review)
 
    bangla_pos, english = split_bengali_english_review(positive_review)
  
    
    
    return np.array(bangla_neg), np.array(bangla_pos)

    
  
    


def read_edited_review_confidence_thr_75():
    positive_review = []
    negative_review = []
    labels = []
    negative_directory = "/Users/russell/Documents/NLP/DownloadedDataFromYoutube/Nov_14/edited_translation_w_label_edited/negative/"
    positive_directory = "/Users/russell/Documents/NLP/DownloadedDataFromYoutube/Nov_14/edited_translation_w_label_edited/positive/"
        
    negative, positive = get_review_data_from_directory(negative_directory, positive_directory)
     
    negative_review = negative_review + negative
    positive_review = positive_review + positive
     
    print("\n 0 Positive: ",len(positive_review))
    print("\n 0 Negative: ",len(negative_review))
   
    print("Positive: ",len(positive_review))
    print("Negative: ",len(negative_review))
    

    
    bangla_neg, english = split_bengali_english_review(negative_review)
 
    bangla_pos, english = split_bengali_english_review(positive_review)
  
    
    
    return np.array(bangla_neg), np.array(bangla_pos)


def get_data_from_excel(filename,sheet_name):
    
    import pandas
    df = pandas.read_csv("/Users/russell/Downloads/finaldataset.csv")
        #print(df)
    data = df.iloc[:,0:1].values.tolist()
    labels = df.iloc[:, 5:6].values.tolist()
    
    #print(data, labels)
    
    #return
       
    negative = 0
    positive = 0
    
    positive_data = []
    negative_data = []
    for i in range(len(data)):
        
        label = str(labels[i]).replace("\'","") 
        if label == "[নিশ্চিত নেতিবাচক]" :
            #print(i, data[i], labels[i])
            negative_data.append(str(data[i]))
            negative += 1
        
        if label == "[নিশ্চিত ইতিবাচক]" :
            #print(i, data[i], labels[i])
            positive_data.append(str(data[i]))
            positive += 1
        
    
    
    print("\n\n-->",negative, positive)
    
    write_reviews("/Users/russell/Downloads/pos_SUST.txt", positive_data)
    write_reviews("/Users/russell/Downloads/neg_SUST.txt", negative_data)
    
    data = negative_data + positive_data
    label = [0]* len(negative_data) + [1]* len(positive_data)
    
    #return data, label

    return np.array(data), np.array(label)
    
    '''
    from pandas import read_excel
    #data = read_excel('/Users/russell/Documents/NLPPaper/Aspect_Sentiment_Analysis/Dataset/Food-Review-2.xlsx', sheet_name = my_sheet_name)
    data = read_excel(filename, sheet_name = sheet_name)
    #data = pd.read_csv("/Users/russell/Documents/NLPPaper/Comments.csv") #text in column 1, classifier in column 2.
    numpy_array = data.values
   # print(numpy_array)
    X = numpy_array[0:,0]
    Y = numpy_array[0:,5]
    
    
    for i in range(10):
        print(X[i], Y[i])
  
    
    
    reviews = []
    ratings = []
    
    negative = []
    positive = []
    neutral = []
    label_neg = []
    label_pos = []
    label_neu = []
    
    x_data = []
    y_label = []
    
    '''


#---- Sentiment Lexicon ------

    
def senticnet_lexicon():
   
    #file = open('/Users/russell/Downloads/senticnet-4.0/senticnet4.txt', 'r') 
    file = open('/Users/russell/Downloads/senticnet-5.0/senticnet5.txt', 'r') 
    for line in file: 
         #elements = line.rstrip().split(' ')[3:]
        (key, val,score) = line.split()
        sentic_net[key] = score
         


def Bing_liu_lexicon():
    file = open('/Users/russell/Documents/NLP/resource/positive.txt', 'r') 
    for line in file: 
        token = line.split()
        key = ''.join(token)
        bing_liu_lexicon[key] = 1
        
    file = open('/Users/russell/Documents/NLP/resource/negative.txt', 'r') 
    for line in file: 
        token = line.split()
        key = ''.join(token)
        bing_liu_lexicon[key] = -1
        


def get_vader_lexicon():
    file = open('/Users/russell/Downloads/vader-lexicon/vader_lexicon.txt', 'r') 
    sentiments = []
    for line in file: 
        token = line.split("\t")
        key = token[0]
        score = token[1]
        print(key, score)
        vader_lexicon[key] = float(score)
        sentiments.append(key + "," + score)
        
    '''
    f = open("/Users/russell/Downloads/vader_word_score.txy","w+")
    for i in range(len(sentiments)):
        f.write(sentiments[i] + "\n")
    '''
        
#--------End----------
def read_sentiment_candidates():
    path = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/Lexicon_Generation/temp_se_en.txt"
    words = []
    with open(path) as file:
        for line in file:
            words.append(line)
          
    path = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/Lexicon_Generation/temp_se.txt"
    words_bn = []
    with open(path) as file:
        for line in file:
            words_bn.append(line)
            
        
    return words_bn, words

    
           

def write_reviews(filename, reviews):
    f = open(filename,"w+")
    for i in range(len(reviews)):
        f.write(reviews[i] + "\n")
        


#--------------- Extract Word-----
def extract_word_from_reviews():
  #bengali_data = "all_results_27293_bengali.txt"
    bengali_data = "all_results_bengali_42627.txt" #"all_results_27293_bengali_old.txt"
    #english_data =  "all_results_27293_english.txt"
    docs = get_reviews(review_directory, bengali_data)
    #count_word(docs)
    
    
#def count_word(docs):
    
    
    vectorizer = CountVectorizer(ngram_range=(1,1), tokenizer=lambda x: x.split())
# tokenize and build vocab
    vectorizer.fit(docs)
    #print('vocabulary: ', vectorizer.vocabulary_)
    
    bag_of_words = vectorizer.transform(docs)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    words_freq = words_freq[:]

    sentiment_words = []
    i = 0
    '''
    for word,freq  in words_freq:
        if freq <10:
           break
        
        print(i, word, freq)
        i += 1
        sentiment_words.append(word)
        
     '''   
    for word,freq  in words_freq:
        if freq > 10:
           continue
       
        if freq <3:
           break
        
        print(i, word, freq)
        i += 1
        sentiment_words.append(word)
        

    write_reviews("/Users/russell/Downloads/temp_se_freq_3_10.txt", sentiment_words)
    #print( words_freq[:500])
    
def extractSentimentWord(sentences, words_bn):  
    from nltk.corpus import sentiwordnet as swn
    count_adj = 0
    lexicons = []
    lexicons_bn = []
    i =0 
    for sentence in sentences:  
        tokens = nlp_english(sentence)
    
        
        #text = word_tokenize(sentence)
        #print("## ##",nltk.pos_tag(text))
        
     
               
        
        for token in tokens: 
            # if not token.is_stop:
            #if  token.dep_ == 'nsubj' or  token.dep_ == 'amod' or token.pos_ == 'ADJ':
            #lexicons.append(token.text)
            #lexicons_bn.append(words_bn[i]) 
            
            if token.dep_ == 'advmod' or token.dep_ == 'compound' or token.pos_ == 'ADV' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'NOUN':
               #print(token.text, token.dep_,  token.pos_,  [child for child in token.children])
               lexicons.append(token.text)
               lexicons_bn.append(words_bn[i])
               
               if token.pos_ == 'VERB':
                  # print(token.text, token.pos_, words_bn[i])
                   count_adj += 1
             
          
                   
        i += 1
            

    bing_liu = 0
    vader = 0
    sentiword = 0
    senticnet_match = 0
    all_words = []
    

    
    for i in range(len(lexicons)):
        word = lexicons[i].strip().lower()
        all_words.append(word)
        
        
        if word in bing_liu_lexicon:
            #print(lexicons_bn[i].strip(), word, bing_liu_lexicon[word])
            #print(word)
            print(lexicons_bn[i].strip())
            bing_liu += 1
        
        
        '''
        if word in vader_lexicon:
            print("Vader: ",lexicons_bn[i].strip(), word, vader_lexicon[word])
            vader += 1
        '''
        
        '''
        if word in sentic_net:
            print("Sentic: ",lexicons_bn[i].strip(), word, sentic_net[word])
            senticnet_match += 1
        '''
            
            
            
        '''
           
        wn_tag = penn_to_wn(word)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue
        
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue
       
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue
 
        # Take the first sense, the most common
        synset = synsets[0]
        polarity_s = swn.senti_synset(synset.name())

        polarity_s = swn.senti_synset(word + "." + wn_tag + "." + "03")
        if abs(polarity_s.pos_score() - polarity_s.neg_score()) > 0.25:
            print("SentiWordNet ", polarity_s)
            sentiword += 1
        '''
        
    return
    

    not_in_english = []
    for word in all_words:
        if word not in vader_lexicon:
            print(word)
            not_in_english.append(word)
            
    write_reviews("/Users/russell/Downloads/non_in_English.txt", not_in_english)
    print(len(lexicons), bing_liu, vader, senticnet_match) 
    
   #write_reviews("/Users/russell/Downloads/sentiment.txt", not_in_english)
    #print(len(lexicons), bing_liu, vader, senticnet_match)  




def penn_to_wn(tag):
    
    
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None



#--------------------- Generate synonyms/Antonyms ---------------
    
def generate_synonyms(directory_name):
    
    synonym_dictionary = {}
    reviews = []
    
    synonym_directory = directory_name + "/with_synonym/" 
    for file_path in os.listdir(synonym_directory):
        file_path = synonym_directory + file_path
        
        if ".txt" not in file_path:
            continue
        
        #i = 0
        print("---->",file_path)
        with open(file_path) as file:
            lines = []
            for line in file:
                line = line.strip()
                lines.append(line)
                
                synonym_dictionary[line] = 1
        
        
    for key in synonym_dictionary:
        reviews.append(key)
      
    write_reviews(directory_name + "/synonym_lexicon.txt", reviews)          
                
    
    
def create_lexicon_from_multiple_language():
    directory_name = "/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/Lexicon_Generation/Bing_Liu/"
    import os
    
    n_lang = 8 # len(os.listdir(directory_name))
    rows, cols = (403, n_lang) 
    multi_lexicon = [['0']*cols]*rows 
    
    j = 0
    
    senti = []
    
    for file_path in os.listdir(directory_name):
        file_path = directory_name + file_path
        
        if ".txt" not in file_path:
            continue
        
        #i = 0
        print("---->",file_path)
        with open(file_path) as file:
            lines = []
            for line in file:
                line = line.strip()
                lines.append(line)
                #print(i,j,line)
                #value = (line + '.')[:-1]
                
            senti.append(lines)
         
        '''
        for i in range(len(lines)):
            multi_lexicon[i][j] = lines[i]
                #i = i +  1
       
        j = j +  1
        '''
    #return
    
    unique_dic = {}
    unique_list = []
    combined = []
    for j in range(403):
        words = ""
        dic = {}
        for i in range(7):
            if senti[i][j] not in dic: 
                words += "," + senti[i][j]
                dic[senti[i][j]] = 1
            if senti[i][j] not in unique_dic:
                unique_dic[senti[i][j]] = 1
                unique_list.append(senti[i][j])
            
             
        print( words)
        combined.append(words)
        
    write_reviews("/Users/russell/Downloads/combined.txt", combined)
    
    print("unique_dic: ", len(unique_dic))
    
    write_reviews("/Users/russell/Downloads/unique_list.txt", unique_list)
        
    return
    
    
    combined = []
    for i  in range(403):
        words = ""
        for j in range(2,3):
            #words += "," + multi_lexicon[i][j]
            print(i,j, multi_lexicon[i][j])
            
        combined.append(words)
        
    write_reviews("/Users/russell/Downloads/combined.txt", combined)
            
#print(file
    
    

'''    
def get_top_words_from_labeled_set(doc):
     
    vectorizer = CountVectorizer(ngram_range=(1,1), tokenizer=lambda x: x.split())
    vectorizer.fit(doc)
    bag_of_words = vectorizer.transform(doc)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
   
    return words_freq
    
    print(words_freq)
'''

#---------------------------------------- Labled/unlabeled data ------   
    
def get_top_words_from_labeled_set(doc):
    
    sentence_tokens = []
    
    print(len(doc))
    
    for i in range(len(doc)):
        tokens = doc[i].split()
        #print(i, tokens)
        sentence_tokens.append(tokens)
        
    
    return sentence_tokens
    
    vectorizer = CountVectorizer(ngram_range=(1,1), tokenizer=lambda x: x.split())
    vectorizer.fit(doc)
    bag_of_words = vectorizer.transform(doc)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
   
    return words_freq
    
    print(words_freq)
    
    
def get_unabeled_data():
    
    path = "/Users/russell/Documents/NLP/DownloadedDataFromYoutube/Nov_14/all_results_bengali_42627.txt"
    reviews = []
    with open(path) as file:
        for line in file:
            tokens = line.split(":")
            reviews.append(tokens[1])
            #print(tokens[1])
            
    return np.array(reviews)
    
    
def assign_label_to_unlabeled_data():
    
    testing_data = get_unabeled_data()
    
    bengali_neg, bengali_pos = read_edited_review_confidence_thr_75()
    bengali_neg_2, bengali_pos_2  = read_final_reviews_data_between_0_75()
   
    bengali_neg = np.concatenate((bengali_neg,bengali_neg_2),axis=0)
    bengali_pos = np.concatenate((bengali_pos,bengali_pos_2),axis=0)
    
    
    data =  np.concatenate((bengali_neg,bengali_pos),axis=0)
    
    label_neg = [0] * len(bengali_neg)
    label_pos = [1] * len(bengali_pos)
    
    label =  np.concatenate((label_neg,label_pos),axis=0)
    
    
    c = list(zip(data, label))
    random.shuffle(c)
    data,  label  = zip(*c)
    
    
    
    num_of_training_data = len(data)
    
    data =  np.concatenate((data,testing_data),axis=0)
    
    tf_idf = TF_IDF()
    
    
    

    data = tf_idf.get_tf_idf(data)
    
    label_train  = label
    

    data_train = data[:num_of_training_data] 
    data_test = data[num_of_training_data:]
    
    '''
    num_of_training_data = 10000
    label_train = label[:num_of_training_data]
    label_test = label[num_of_training_data: ]
    '''
    
    classifier =  Logistic_Regression_Classifier()
    prediction = classifier.predict(data_train, label_train, data_test)
    
    
    print(len(testing_data), len(prediction))
    
    negative_rev = []
    positive_rev = []
    for i in range(len(testing_data)):
        if prediction[i] == 0:
            negative_rev.append(testing_data[i])
        else:
            positive_rev.append(testing_data[i])
        
    print(len(negative_rev), len(positive_rev))
    
    
    calculate_PMI_from_labeled_data(negative_rev, positive_rev,"/Users/russell/Downloads/unabeled_temp.txt")
    
    
    '''
    performance = Performance() 
    precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)
    print("\n\nEnglish P-R-F1-Acc: ",round(precision,3), round(recall,3)   , round(f1_score, 3), round(acc,3))
    '''
    

def calculate_PMI_from_labeled_data(bengali_neg, bengali_pos, output_file):
    
    sentence_tokens_neq = get_top_words_from_labeled_set(bengali_neg)
    sentence_tokens_pos = get_top_words_from_labeled_set(bengali_pos)
    
    #return
   
    #print(freq_neq[:100])
    #print(freq_pos[:100])
    
    
    input_dict = {}
    input_dict['neg'] = sentence_tokens_neq
    input_dict['pos'] = sentence_tokens_pos
    
    '''
    input_dict = {
    "label_a": [
        ["I", "aa", "aa", "aa", "aa", "aa"],
        ["bb", "aa", "aa", "aa", "aa", "aa"],
        ["I", "aa", "hero", "some", "ok", "aa"]
    ],
    "label_b": [
        ["bb", "bb", "bb"],
        ["bb", "bb", "bb"],
        ["hero", "ok", "bb"],
        ["hero", "cc", "bb"],
    ],
    "label_c": [
        ["cc", "cc", "cc"],
        ["cc", "cc", "bb"],
        ["xx", "xx", "cc"],
        ["aa", "xx", "cc"],
    ]
    }
    '''
    from DocumentFeatureSelection import interface
    x = interface.run_feature_selection(input_dict, method='pmi', use_cython=True).convert_score_matrix2score_record()
        
    
    
   # y = interface.run_feature_selection(input_dict, method='bns', use_cython=True).convert_score_matrix2score_record()
    
    '''
    from DocumentFeatureSelection.common.data_converter import DataCsrMatrix
    from DocumentFeatureSelection.bns import bns_python3
    from DocumentFeatureSelection.common import data_converter
    
    data_csr_matrix = data_converter.DataConverter().convert_multi_docs2document_frequency_matrix(
            labeled_documents=input_dict,
            n_jobs=5
        )
    assert isinstance(data_csr_matrix, DataCsrMatrix)
    csr_matrix_ = data_csr_matrix.csr_matrix_
    n_docs_distribution = data_csr_matrix.n_docs_distribution

    result_bns = bns_python3.BNS().fit_transform(X=csr_matrix_,
                                    y=None,
                                    unit_distribution=n_docs_distribution,
                                    use_cython=True)

    print(x[:10])
    print(result_bns[:10])
    
    '''
    #return
    #list_of_words = []
    
    list_of_words = set()
    word_dic = {}
    for i in range(5000): #1300 used for labeled
        word = x[i]
        #print(word)
        #list_of_words.append(word['feature'])
        list_of_words.add(word['feature'])
        #print(word['feature'], word['label'], "    ",word['frequency'], round(word['score'],6))
        key = word['feature'] + "_" + word['label']
        #print(key)
        word_dic[key] = str(word['frequency']) + "_" + str(round(word['score'],6))
    
    
    print("~~~~~~~~~~~~~~")
    top_words = []
    negative = []
    for word in list_of_words:
        key_1 = word + "_neg"
        key_2 = word + "_pos"
        if key_1 not in word_dic or key_2 not in word_dic:
            continue
        freq_neg = float(word_dic[key_1].split("_")[0])
        freq_pos = float(word_dic[key_2].split("_")[0])
        
        diff = float(freq_pos) - float(freq_neg)
        
        diff_ratio = abs(diff/float(freq_neg + freq_pos))
        
        if  diff_ratio > 0.50:
            #print(word) #, diff)
            top_words.append(word)
        #freq_neg, freq_pos,
        
    print("Number of word: ",len(top_words))
    
    
  
    #return
    
    
    lexicon_from_english= []
    
    filename= "/Users/russell/Documents/NLP/Paper-4/resources/unique_list.txt"
    with open(filename) as text:
        for line in text:
            lexicon_from_english.append(line.strip())
            
            
    lexicon_from_labled = []
    filename= "/Users/russell/Documents/NLP/Paper-4/resources/lexicon_from_training_set_11807.txt"
    with open(filename) as text:
        for line in text:
            lexicon_from_labled.append(line.strip())
            
            
    print("\n\n--- ")
    count_found = 0
    count_not_found = 0
    
    unlabeled_lexicon = []
    for word in top_words:
        if word not in lexicon_from_english and word not in lexicon_from_labled:
            #print(word)
            count_not_found += 1
            unlabeled_lexicon.append(word)
            
        else:
            #print(word)
            count_found += 1
    
          
    print("Not found, found",count_not_found, count_found)
    write_reviews(output_file, unlabeled_lexicon)
    
    
  
#---------------------------Lexicon Coverage---

def vader_lexicon_bengali():
    #file = open('/Users/russell/Documents/NLP/Paper-4/Lexicons/vader-lexicon/bengali_vader.txt', 'r') 

    file = open('/Users/russell/Documents/NLP/Paper-4/Lexicons/vader-lexicon/bengali_vader_emo_removed.txt', 'r') 
    #file = open("/Users/russell/Documents/NLP/Paper-4/Lexicons/vader-lexicon/bengali_vader_final_list.txt", 'r') 
   
    
    vader_lexicon_bengali = {}
    index = 0
    for line in file: 
        token = line.split(",")
        
        
        #print("???????? ", len(token), token)
        if len(token) == 1:
            
            token = line.split()
            key = token[0]
            score = token[-1]
            #print("---> ", token, key, score)
            # key.replace(",", "")
            
            
            '''
            if key.strip() == "না":
                continue
            '''
                
            
            vader_lexicon_bengali[key] = float(score)
        
            index += 1
            
            continue
            
            
        if len(token) == 3:
            #print(token)
            
            key = token[0]
            score = token[1] + "." + token[2]
            score = float(score.strip())
            #print(key, score)
             # key.replace(",", "")
       
            vader_lexicon_bengali[key] =  score #float(score)
        
            index += 1
            
            
            continue
            
            
            
            
            
        
        key = token[0]
        score = token[1]
        score = float(score.strip())
        
        
        '''
        if abs(float(score)) < 1.0:
            #print("#####")
            continue 
        '''
       
        #print(key, score)
       # key.replace(",", "")
       
        vader_lexicon_bengali[key] =  score #float(score)
        
        index += 1
        
    
    file = open("/Users/russell/Documents/NLP/Paper-4/Lexicons/vader-lexicon/bengali_vader_final_final_list.txt", 'w') 
    for key in vader_lexicon_bengali: 
       file.write(str(key) + "," + str(vader_lexicon_bengali[key])  + "\n")
    
    print(len(vader_lexicon_bengali))
    return vader_lexicon_bengali
        
def Bing_liu_lexicon_bengali():
    
    bengali_bing_liu = {}
    
    file = open('/Users/russell/Documents/NLP/Paper-4/Lexicons/Bing Liu/bengali_negative.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.strip() # ''.join(token)
        bengali_bing_liu[key] = -1
        #print(key)
        
    file = open('/Users/russell/Documents/NLP/Paper-4/Lexicons/Bing Liu/bengali_positive.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.strip()
        bengali_bing_liu[key] = 1
        #print(key)
        
    return bengali_bing_liu


def AFINN_lexicon_bengali():
    
    
    AFINN_lexicon = {}
    
    file = open('/Users/russell/Documents/NLP/Paper-4/Lexicons/AFINN_BN/AFINN_bengali_words_final.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.split()[0] # ''.join(token)
        score = line.split()[-1]
        #print(key, score)
        AFINN_lexicon[key] = float(score) # 1
        #print(key)
        
   
        
    return AFINN_lexicon


# BengSentiLex
def SS_lexicon():
    
    bengali_lexicon_ss = {}
    '''
    file = open('/Users/russell/Documents/NLP/Paper-4/Lexicons/Bing Liu/bengali_negative.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.strip()
        bengali_lexicon_ss[key] = -1
    ''' 
    '''
    file = open('/Users/russell/Documents/NLP/Paper-4/resources/unique_list_corrected.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.strip()
        #print(key)
        bengali_lexicon_ss[key] = 1
        
        
    file = open('/Users/russell/Documents/NLP/Paper-4/resources/Lexicon_Labeled/synonym_lexicon.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.strip()
        #print(key)
        bengali_lexicon_ss[key] = 1
        
    file = open('/Users/russell/Documents/NLP/Paper-4/resources/Lexicon_Unlabeled/synonym_lexicon_corrected.txt', 'r') 
    for line in file: 
        #token = line.split()
        key =  line.strip()
        #print(key)
        bengali_lexicon_ss[key] = 1
        
    
    
    
    words = []
    #f = open("/Users/russell/Documents/NLP/Paper-4/resources/full_list_from_3_sets.txt","w+")
    for key in bengali_lexicon_ss:
        words.append(key)
        #f.write(key + "\n")
        
    words = sorted(words)
    f = open("/Users/russell/Documents/NLP/Paper-4/resources/full_list_from_3_sets.txt","w+")
    for i in range(len(words)):
        f.write(words[i] + "\n")
    '''
    
    #file = open("/Users/russell/Documents/NLP/Paper-4/resources/full_list_from_3_edited.txt", 'r') 
    
    #file = open("/Users/russell/Documents/NLP/Paper-4/resources/added_manually_to_make_1000/unique_list_combined.txt", 'r')
    
    #file = open("/Users/russell/Documents/NLP/Paper-4/resources/added_manually_to_make_1000/neg_pos-auto-manu-all/unique_list_all_auto_manu.txt", 'r')
    
    file = open("/Users/russell/Documents/NLP/Paper-4/resources/added_manually_to_make_1000/neg_pos-auto-manu-all/unique_list_negative_auto_manu.txt", 'r')
    for line in file: 
        #token = line.split()
        key =  line.strip()
        bengali_lexicon_ss[key] = -1
        
    
    #file = open("/Users/russell/Documents/NLP/Paper-4/resources/added_manually_to_make_1000/neg_pos-auto-manu-all/unique_list_all_auto_manu.txt", 'r')
    
    file = open("/Users/russell/Documents/NLP/Paper-4/resources/added_manually_to_make_1000/neg_pos-auto-manu-all/unique_list_positive_auto_manu.txt", 'r')
  
    for line in file: 
        #token = line.split()
        key =  line.strip()
        bengali_lexicon_ss[key] = 1
    
        
    print("++++++++ ",len(bengali_lexicon_ss))
        
    return bengali_lexicon_ss

#---------------------- ----Lexica ---------
   
def coverage_in_dictionary(lexicon_dictionary):
    
    
    
    
    #file = open('/Users/russell/Documents/NLP/Paper-4/resources/training/negative_aa.txt', 'r') 
  
    
    file = open('/Users/russell/Documents/NLP/Paper-4/resources/testing/negative_test.txt', 'r') 
    
    #file = open('/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/socian-bangla-sentiment-dataset-labeled-master/socian-bangla-sentiment-dataset-labeled-master/bangla_neg.txt', 'r') 
    
    #file = open("/Users/russell/Downloads/neg_social_new.txt",'r') 
    
    #file = open("/Users/russell/Downloads/neg_mendely.txt", 'r') 
    
    
    pos_found = 0
    neg_found = 0
    
    sentiment_match = {}
    
    #print("@@@@@@")
    
              
    not_found_reviews = []
    
    reviews_with_polarity = 0
    
    num_of_neg_line = 0
    i = 0
    for line in file: 
        tokens = line.split()
        does_polarity_found = 0
        for token in tokens:
            if token in lexicon_dictionary:
                #print(i,token)
                does_polarity_found = 1
                if token in sentiment_match:
                    sentiment_match[token] += 1
                else:
                     sentiment_match[token] = 1
                neg_found += 1
                #break
        if does_polarity_found == 1:
            reviews_with_polarity += 1
        else:
            #print("Neg: ",line)
            not_found_reviews.append(line)
        num_of_neg_line += 1
        #if num_of_neg_line == 5000:
           # break
        i += 1
    
         
    
    #file = open('/Users/russell/Documents/NLP/Paper-4/resources/training/positive_aa.txt', 'r') 
    #file = open('/Users/russell/Documents/NLP/Paper-4/resources/testing/positive_test.txt', 'r') 
    file = open('/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/socian-bangla-sentiment-dataset-labeled-master/socian-bangla-sentiment-dataset-labeled-master/bangla_pos.txt', 'r') 
    #file = open("/Users/russell/Downloads/pos_social_new.txt",'r') 
    #file = open("/Users/russell/Downloads/pos_mendely.txt", 'r') 
    
    
    
    i = 0
    num_of_pos_line = 0
    for line in file: 
        tokens = line.split()
        does_polarity_found = 0
        for token in tokens:
            if token in lexicon_dictionary:
                #print(i , token)
                does_polarity_found = 1
                if token in sentiment_match:
                    sentiment_match[token] += 1
                else:
                     sentiment_match[token] = 1
                pos_found += 1
                #break
        if does_polarity_found == 1:
            reviews_with_polarity += 1
        else:
            #print("Pos: ",line)
            not_found_reviews.append(line)
        num_of_pos_line += 1
        #if num_of_pos_line == 5000:
           # break
        i += 1

    import operator
    sorted_dic = sorted(sentiment_match.items(), key=operator.itemgetter(1),reverse=True)
    
    
    ##for key in sorted_dic:
        #print(key)
    
    
    
    '''
    file = open("/Users/russell/Downloads/drama_nifound_pos_nee.txt", 'w') 
    for key in sorted_dic: 
       file.write(str(key) + "\n")
    '''

    
    
    '''
    file = open("/Users/russell/Downloads/not_found.txt", 'w') 
    for review in not_found_reviews: 
       file.write(review  + "\n")
    '''
    
    
    #print("Word level polarity: ", neg_found, pos_found)
            
    print("#polarity_found: ", reviews_with_polarity, " Total", num_of_neg_line, num_of_pos_line, num_of_neg_line + num_of_pos_line  )
  
     
    return neg_found, pos_found


def integrate_in_lexicon_based_classifier(lexicon_dictionary):
    
    #file = open('/Users/russell/Documents/NLP/Paper-4/resources/training/negative_aa.txt', 'r') 
  

    file = open('/Users/russell/Documents/NLP/Paper-4/resources/testing/negative_test.txt', 'r') 
    
    #file = open('/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/socian-bangla-sentiment-dataset-labeled-master/socian-bangla-sentiment-dataset-labeled-master/bangla_neg.txt', 'r') 
    
    #file = open("/Users/russell/Downloads/neg_social_new.txt",'r') 
    
    #file = open("/Users/russell/Downloads/neg_mendely.txt", 'r') 
    
    
    pos_found = 0
    neg_found = 0
    
    sentiment_match = {}
    not_found_reviews = []
    

    total_num_of_neg_review = 0
    i = 0
    num_of_negative = 0
    for line in file: 
        tokens = line.split()
        
        polarit_score = 0
        for token in tokens:
            if token in lexicon_dictionary:
                #print(i,token)
                polarit_score += lexicon_dictionary[token]
                
        if polarit_score < 0:
            num_of_negative += 1
               
        total_num_of_neg_review += 1
        
        if i == 5204:
            break
        
        i += 1
        
    
    
    file = open('/Users/russell/Documents/NLP/Paper-4/resources/testing/positive_test.txt', 'r') 
    
    #file = open('/Users/russell/Documents/NLP/Paper-3-Machine_Translation_SSentiA/socian-bangla-sentiment-dataset-labeled-master/socian-bangla-sentiment-dataset-labeled-master/bangla_pos.txt', 'r') 
    #file = open("/Users/russell/Downloads/pos_social_new.txt",'r') 
    
    total_num_of_pos_review = 0
    num_of_positive = 0
    for line in file: 
        tokens = line.split()
        
        polarit_score = 0
        for token in tokens:
            if token in lexicon_dictionary:
                #print(i,token)
                polarit_score += lexicon_dictionary[token]
                
        if polarit_score > 0:
            num_of_positive += 1
               
        total_num_of_pos_review += 1
    
    
    total = total_num_of_neg_review + total_num_of_pos_review
    
    correct = num_of_negative + num_of_positive
    
    print("\n")
    
    print("Neg --->", num_of_negative, total_num_of_neg_review, num_of_negative / total_num_of_neg_review)
   
    print("Pos --->", num_of_positive, total_num_of_pos_review, num_of_positive / total_num_of_pos_review)
    
    
    print("Total --->", correct, total, correct/total)
    
   
    

# Call this function to check the coverage of various lexicons    
def check_lexicon_coverage():
    
    
   
    
    
    print("\nAFINN")
    afin = AFINN_lexicon_bengali()
    integrate_in_lexicon_based_classifier(afin)
    
    
    print("\n\nBing Liu")
    bing_liu = Bing_liu_lexicon_bengali()
    integrate_in_lexicon_based_classifier(bing_liu)
    
    
    
    print("\n\nVADER")
    vader = vader_lexicon_bengali()
    integrate_in_lexicon_based_classifier(vader)
    
   
    
    print("\n\nBengSentiLex")
    ss = SS_lexicon()
    integrate_in_lexicon_based_classifier(ss)
    
    
    
    
    
    return
    vader = vader_lexicon_bengali()
    
    #return
    bing_liu = Bing_liu_lexicon_bengali()
    ss = SS_lexicon()
    #vader = vader_lexicon_bengali()
    #print(len(vader), len(bing_liu), len(ss))
    #return
    
    
    print("AFINN----")
   
    neg_afin, pos_afin = coverage_in_dictionary(afin)
    print(neg_afin, pos_afin)
    #return

    
    
    print("VADER ----")
    neg_vader, pos_vader = coverage_in_dictionary(vader)
    print(neg_vader, pos_vader)
    #return
    

    
    #return
    
    
    print("Bing ----")
    neg_bing, pos_bing = coverage_in_dictionary(bing_liu)
    print(neg_bing, pos_bing)
    #return
    
   
    print("SS ----")
    neg_ss, pos_ss = coverage_in_dictionary(ss)
    print(neg_ss, pos_ss)
    return
    
                
            
            
    

            
    print(neg_vader, neg_bing, neg_ss)
    print(pos_vader, pos_bing, pos_ss)
    
   # 
   # print(len(vader), len(bing_liu))
    
    
#---------------- SML -------------    
def apply_SML_classifier(data, label):   

    c = list(zip(data, label))
    random.shuffle(c)
    data, label  = zip(*c)
    print("Total Data all directory: ",len(label), type(label), label[1])
    
    
    label = np.asarray(label)
    #label=label.astype('int')
       
    tf_idf = TF_IDF()
    data = tf_idf.get_tf_idf(data)



    num_of_fold = 10
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_of_fold)
    
    #X_train_english, Y_train_english  = get_sampled_data(X_train_english, Y_train_english)
    
    
    total_bengali_f1 = 0
    total_english_f1 = 0
    total_bengali_acc = 0
    total_english_acc = 0
    total_bengali_precison = 0
    total_english_precison = 0
    total_bengali_recall = 0
    total_english_recall = 0
    
    #prediction_english_all = []
    #prediction_bengali_all = []
    
    prediction_english_all = np.array([]) 
    prediction_bengali_all = np.array([]) 
    

    for train_index, test_index in kf.split(data, label):

        classifier = Logistic_Regression_Classifier() 
        
        label_train, label_test = label[train_index], label[test_index]
        data_train, data_test = data[train_index], data[test_index]
       
        prediction = classifier.predict(data_train, label_train, data_test)
    
        performance = Performance() 
    
        
        
        precision,  recall, f1_score, acc = performance.get_results(label_test, prediction)
    
        print("Bengali F1#",f1_score)
        total_bengali_f1 += f1_score
        total_bengali_acc += acc
        total_bengali_precison  += precision
        total_bengali_recall += recall
        
        #print
    
    
    print("Overall")
    #print("\n\nEnglish P-R-F1-Acc: ",round(total_english_precison/num_of_fold,3), round(total_english_recall/num_of_fold,3)   , round(total_english_f1/num_of_fold, 3), round(total_english_acc/num_of_fold,3))
    print("\nBengali: P:R:F1-Acc", round(total_bengali_precison/num_of_fold,3), round(total_bengali_recall/num_of_fold,3) , round(total_bengali_f1/num_of_fold,3), round(total_bengali_acc/num_of_fold, 3))
    print("\n\n------")
    
    
    
#---------- Main ---------
            
   
'''
import polyglot
from polyglot.text import Text, Word

text = Text("The movie was really good.")

for w in text:
    print("{:<16}{:>2}".format(w, w.polarity))

return
'''
   
def main():
    
    #get_mendely_paper()
    #return
    
    check_lexicon_coverage()
    return

    get_reviews_social_media_news_paper()
    return
    
    
    data, label = get_data_from_excel("/Users/russell/Documents/NLP/Paper-4/resources/finaldataset.xlsx", "finaldataset")
    apply_SML_classifier(data, label)
    return 
    '''
    generate_synonyms("/Users/russell/Documents/NLP/Paper-4/resources/Lexicon_Labeled")
    return
    assign_label_to_unlabeled_data()
    return 
    '''
    
   
    #get_vader_lexicon()
    #return
    
    '''
    from afinn import Afinn
    afinn = Afinn()
    print(afinn.score('Hate'))

    return
    '''    
        


    '''
    extract_word_from_reviews()
    return
    '''
    
    
    
    bengali_neg, bengali_pos = read_edited_review_confidence_thr_75()
    bengali_neg_2, bengali_pos_2  = read_final_reviews_data_between_0_75()
   

    
    bengali_neg = np.concatenate((bengali_neg,bengali_neg_2),axis=0)
    bengali_pos = np.concatenate((bengali_pos,bengali_pos_2),axis=0)
    
    
    #write_reviews("/Users/russell/Downloads/negative_aa.txt", bengali_neg)
    #write_reviews("/Users/russell/Downloads/positive_aa.txt", bengali_pos)
    
    #print(len(bengali_neg), len(bengali_pos))
    
    
    #return 
    
    calculate_PMI_from_labeled_data(bengali_neg, bengali_pos, "/Users/russell/Downloads/temp_labeled.txt")
    
    
    return
    create_lexicon_from_multiple_language()
    return
    #generate_lexicon()
    Bing_liu_lexicon()
    vader_lexicon()
    senticnet_lexicon()
    
    words_bn, sentiment_candidates = read_sentiment_candidates()
    extractSentimentWord(sentiment_candidates, words_bn)

if __name__== main():
        main()