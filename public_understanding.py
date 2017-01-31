import time
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import numpy as np
from pymongo import MongoClient
stoplist = stopwords.words('english')

#calculating the cosine similarity between the tweet and title as well as abstract of the research paper.
def cosine_similarity(data_given,index_given,dictionary_given,lda_given):
    vec_bow = dictionary_given.doc2bow(data_given.lower().split())
    vec_lda = lda_given[vec_bow]

    #calculating the cosine similarity
    sims = index_given[vec_lda]
    a = list(enumerate(sims))
    lis = []
    lis.append((a[0][1]*float(100)))
    lis.append((a[1][1]*float(100)))
    return lis

#this function will calculate the public understanding of each and every tweet with respect to abstract and title of a research article.
def best_function(tweets_list,index_function,dictionary_function,lda_function):
    tweets_count_in_file = 0
    tweets_abstract_per_file = 0
    tweets_title_per_file = 0
    returning = []
    for data_g in tweets_list:
        tweets_count_in_file += 1
        per = cosine_similarity(str(data_g),index_function,dictionary_function,lda_function)
        tweets_abstract_per_file += float(per[0])
        tweets_title_per_file += float(per[1])
    if tweets_count_in_file == 0:
        tweets_count_in_file = 1
    tweets_abstract_per_file = tweets_abstract_per_file/tweets_count_in_file
    tweets_title_per_file = tweets_title_per_file/tweets_count_in_file
    returning.append(tweets_abstract_per_file)
    returning.append(tweets_title_per_file)
    return returning
def file_counting(count):
    con = 0
    for i in count:
        if i == 0:
            count[con] = 1
        con+=1
    return count
if __name__=="__main__":
        files_count = np.zeros(6)
        social_media_abstracts_per = np.zeros(6)
        social_media_titles_per = np.zeros(6)
        number_of_files = 0
        client = MongoClient()
        db = client['mycustomers']
        lower_bound = -15151
        higher_bound = 0
        for i in range(0,100):
            lower_bound += 15151
            higher_bound += 15151

            #querying from the mongo database. Here the entire data set will not be read into the memory at a time, instead some portion
            # of the data set is read and it is removed from the memory before reading next part of the data set.
            d = db.altmetric_data.find({'$and':[{'primary_key':{'$gt':lower_bound}},{'primary_key':{'$lte':higher_bound}}]},no_cursor_timeout = True)
            for data in d:
                number_of_files += 1
                paper_title = str(data['title'][0])
                paper_title = paper_title.replace('.','')
                paper_title = paper_title.replace(',','')
				#creating super text.
                texts = []
                supertext_main = []
                #removing the stop words
                for word in data['abstract'][0].split():
                    if word.lower() not in stoplist:
                        texts.append(word.lower())
                supertext_main.append(texts)
                texts1 = []
                for word in data['title'][0].split():
                    if word.lower() not in stoplist:
                        texts1.append(word.lower())
                supertext_main.append(texts1)
				#preparing dictionary for the corpus.
                dictionary_main = corpora.Dictionary(supertext_main)
				#creating the corpus form dictinary and super text.
                corpus_main = [dictionary_main.doc2bow(text) for text in supertext_main]
				#applying lda on the corpus.
                lda_main = models.LdaModel(corpus_main,id2word=dictionary_main, num_topics = 2)
                #preparing matrix to measure the cosine similarity between title,abstract with social networking tweet.
				index_main = similarities.MatrixSimilarity(lda_main[corpus_main])

                #checking for the twitter tweets in the data.
                if 'twitter' in data:
                    percen = best_function(data['twitter'],index_main,dictionary_main,lda_main)
                    social_media_abstracts_per[0] += float(percen[0])
                    social_media_titles_per[0] += float(percen[1])
                    files_count[0]+=1

                #checking for the facebook tweets in the data.
                if 'facebook' in data:
                    percen = best_function(data['facebook'],index_main,dictionary_main,lda_main)
                    social_media_abstracts_per[1] += float(percen[0])
                    social_media_titles_per[1] += float(percen[1])
                    files_count[1]+=1

                #checking for the blog tweets in the data.
                if 'blogs' in data:
                    percen = best_function(data['blogs'],index_main,dictionary_main,lda_main)
                    social_media_abstracts_per[2] += float(percen[0])
                    social_media_titles_per[2] += float(percen[1])
                    files_count[2]+=1

                #checking for the googleplus tweets in the data.
                if 'googleplus' in data:
                    percen = best_function(data['googleplus'],index_main,dictionary_main,lda_main)
                    social_media_abstracts_per[3] += float(percen[0])
                    social_media_titles_per[3] += float(percen[1])
                    files_count[3]+=1

                #checking for the wikipedia tweets in the data.
                if 'wikipedia' in data:
                    percen = best_function(data['wikipedia'],index_main,dictionary_main,lda_main)
                    social_media_abstracts_per[4] += float(percen[0])
                    social_media_titles_per[4] += float(percen[1])
                    files_count[4]+=1

                #checking for the news tweets in the data.
                if 'news' in data:
                    percen = best_function(data['news'],index_main,dictionary_main,lda_main)
                    social_media_abstracts_per[5] += float(percen[0])
                    social_media_titles_per[5] += float(percen[1])
                    files_count[5]+=1
                print('Done with '+str(number_of_files) +' files')
            #clearing the cursor.
            d.close()
			
        files_count = file_counting(files_count)
        after_division_abs_per = np.zeros(6)
        after_division_tit_per = np.zeros(6)
        #calculating the percentage of level of undertanding of social networking communities. The result will be in the third parameter
        np.divide(social_media_abstracts_per,files_count,after_division_abs_per)
        np.divide(social_media_titles_per,files_count,after_division_tit_per)
