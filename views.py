from django.shortcuts import render

# Create your views here.
from django.conf.urls import url
from django.db.models import Prefetch
from django.core.paginator import Paginator, InvalidPage, EmptyPage
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import F
from django.db.models import Avg
from .models import *
from django.http import HttpResponse,HttpRequest, Http404, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.views import View
from django.views.generic import TemplateView





######################################################################################################################

## To upload the data in Django models from a json file
  import json

  with open("yelp_nyc_review.json", "r") as ins: ## Yelp_NYC data 
    array = []
    for line in ins:
      array.append(line)



  dicts = []
  for line in array:
    dicts.append(json.loads(line))
  
  for k in range(len(dicts)):
    user_vote=(dicts[k]['votes'])
    user_i=(dicts[k]['user_id'])
    review_i=(dicts[k]['review_id'])
    user_ratings=(dicts[k]['stars'])
    reviews_date=(dicts[k]['date'])
    user_reviews=(dicts[k]['text'])
    business_id=(dicts[k]['business_id'])



  

    m=Yelp_NYC(user_id=user_i,user_votes=user_vote,review_id=review_i,user_ratings=user_ratings,reviews_date=reviews_date,user_reviews=user_reviews,buisness_id=business_id)
    m.save()
    ## User context is the model with their respected fields as mentioned.


  
 ############################################################################################################################### 

######## K-means clusetring


  
  label_id =[]

  
  user_list =[]

  global_list_features = []

  for i in Yelp_NYC.objects.all():
    text = i.user_reviews.encode('utf-8') 
    sent_text = nltk.sent_tokenize(text)
    print sent_text
    global_list_features1 = []
    sentences_review = []
    for sentence in sent_text:
      list_features_sentence = []
      
      sentences_review.append(sentence)
      tokenized_text = nltk.word_tokenize(sentence)
      tagged = nltk.pos_tag(tokenized_text)
      
      total_words = log(len(tagged)+1)
      nouns = [token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('N')]
      nouns = log(len(nouns)+1)
      verbs = [token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('V')]
      verbs = log(len(verbs)+1)
      verbs_past_1 = len([token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('VBD')])
      verbs_past_2 = len([token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('VBN')])
      verb_past = verbs_past_1 + verbs_past_2
      verb_past = log(verb_past+1)
      try:

        ProRatio = (nouns*1.0)/total_words

      except ZeroDivisionError:
    
        VRatio = 0  
    
      print ProRatio
      list_features_sentence.append(ProRatio)
      list_features_sentence.append(verb_past)
      list_features_sentence.append(verbs)
      #list_features_sentence.append(nouns)
      list_features_sentence.append(total_words)
      global_list_features1.append(list_features_sentence)

    X = np.array((global_list_features1))

    print X
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print kmeans.labels_
    
    #for i in  kmeans.labels_:
      #label_id.append(i)
    
    dictionary = dict(zip(sentences_review, kmeans.labels_))
    gen =[]
    spec = []
    for key, value in dictionary.iteritems():

      
      if value == 0:
        gen.append(key)
        
      if value == 1:
        spec.append(key)

    i.specific_sentences = spec
    i.generic_sentences = gen           
    i.save()  

## Based on K-means clustering, reviews are classified into specific and generic reviews 
###########################################################################################################################


## Filtering the document for target user/item

def documentfilter():
  from nltk.corpus import stopwords
  f=open('filterdoc.txt','w')
  stop = set(stopwords.words('english'))
  filein=open('doc.txt','r')
  stop = set(stopwords.words('english'))
  
  k = filein.readlines()
  for item in k:
    print ([i for i in item.lower().split() if i not in stop])
    b=(str([i for i in item.lower().split() if i not in stop]))
  
    f.write(b)
    f.write('\n')

  f.close()
  f1 =open('newfilterdoc.txt','w')
  with open('filterdoc.txt') as filein:

    for line in filein:
      line=line.replace("u'","")
      line=line.replace("'","")
      line = line.replace("?","")
      line = line.replace("(","")
      line = line.replace(")","")
      line=line.replace(",","")
      line=line.replace('"',"")
      line=line.replace(".","")
      line=line.replace("!","")
      line=line.replace("-","")
      line=line.replace("[","")
      line=line.replace("]","")

      print line
      f1.write(line)
      

  f1.close()
  
######################################################################################################################

## Filtering the documnets for the most similar users/items

def documentfilter1():
  from nltk.corpus import stopwords
  f=open('filterdoc1.txt','w')
  stop = set(stopwords.words('english'))
  filein=open('doc1.txt','r')
  stop = set(stopwords.words('english'))
  
  k = filein.readlines()
  for item in k:
    print ([i for i in item.lower().split() if i not in stop])
    b=(str([i for i in item.lower().split() if i not in stop]))
  
    f.write(b)
    f.write('\n')

  f.close()
  f1 =open('newfilterdoc1.txt','w')
  with open('filterdoc1.txt') as filein:

    for line in filein:
      line=line.replace("u'","")
      line=line.replace("'","")
      line = line.replace("?","")
      line = line.replace("(","")
      line = line.replace(")","")
      line=line.replace(",","")
      line=line.replace('"',"")
      line=line.replace(".","")
      line=line.replace("!","")
      line=line.replace("-","")
      line=line.replace("[","")
      line=line.replace("]","")

      print line
      f1.write(line)
      

  f1.close()
  
##############################################################################################################

## Embedding generations for the target user/item items reviews


def embeddingsgeneration():

  import preprocess
  from scipy import sparse
  import numpy as np
  import os
  import pandas as pd
  from scipy import sparse
  import CoEmbedding

  DATA_DIR_Results = 'Provide the directory name'
  DATA_DIR = 'Provide the directory name'
  doc_pt = 'Provide the directory name'
  dwid_pt = DATA_DIR+'Provide the text file name'

  dwmatrix_pt = DATA_DIR+'Provide the csv file name'
  voca_pt = DATA_DIR+'Provide the text file name'
  batch_size = 1000
  window_size = 5
  
  n_docs,w2id = preprocess.indexFile(doc_pt,dwid_pt,dwmatrix_pt,batch_size,window_size)
  
  n_words = len(w2id)

  print('n(d)=', n_docs, 'n(w)=', n_words)

  preprocess.write_w2id(voca_pt,w2id)

  matrixD = preprocess.load_data(dwmatrix_pt, (n_docs, n_words))

  print matrixD

  start_idx = list(range(0, n_docs, batch_size))

  end_idx = start_idx[1:] + [n_docs]

  matrixW = sparse.csr_matrix((n_words, n_words), dtype='float32')

  for lo, hi in zip(start_idx, end_idx):
    matrixW = preprocess._matrixw_batch(lo, hi, matrixW, n_words)
    print(float(matrixW.nnz) / np.prod(matrixW.shape))


  np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'), matrixW.data) # co-occurence matrix of each word with total no word frequency with another word
  np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'), matrixW.indices) # gives an array about the position on which we have data
  np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'), matrixW.indptr) # gives an array describes the starting postion of data in each coloumn in matrix

    
  print(matrixD.shape, matrixW.shape)

  
  tp = pd.read_csv(dwmatrix_pt)

  rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])

  matrixD = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=(n_docs, n_words))
  import main
  matrixD = main.tfidf(matrixD,n_docs, normalize=True)

  data = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'))
  indices = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'))
  indptr = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'))
  matrixW = sparse.csr_matrix((data, indices, indptr), shape=(n_words, n_words))

  print(matrixD.shape, matrixW.shape)
  print(float(matrixD.nnz) / np.prod(matrixD.shape))
  print(float(matrixW.nnz) / np.prod(matrixW.shape))

  count = np.asarray(matrixW.sum(axis=1)).ravel() # gives the totla sum of each row
  n_pairs = matrixW.data.sum() # total sum of all matrix values

#constructing the SPPMI matrix
  M = matrixW.copy() # copy matrix into another one.
  for i in range(n_words):                                        # to create a point wise mutual information
    lo, hi, d, idx = main.get_row(M, i)
    M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
  

  print(max(M.data))
  print(M[0,0])
  
  M.data[M.data < 0] = 0
  M.eliminate_zeros()
  print(float(M.nnz) / np.prod(M.shape))

  k_ns = 1
  M_ns = M.copy()
  if k_ns > 1:
    offset = np.log(k_ns)
  else:
    offset = 0.
  
  M_ns.data -= offset
  M_ns.data[M_ns.data < 0] = 0
  M_ns.eliminate_zeros()
  print((np.absolute(M_ns).sum())/np.prod(M_ns.shape))
  vocab_pt = voca_pt
  n_embeddings = 50
  max_iter = 20
  n_jobs = 8
  c0 = 1
  c1 = 1
  K = 20
  lam_sparse_d = 1e-2
  lam_sparse = 1e-7
  lam_d = 0.5
  lam_w = 1
  lam_t = 50
  save_dir = os.path.join(DATA_DIR, 'results_parallel')
  sejwal = CoEmbedding.CoEmbedding(n_embeddings=n_embeddings, K=K, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs, 
            random_state=98765, save_params=True, save_dir=save_dir, verbose=True, 
            lam_sparse_d=lam_sparse_d, lam_sparse=lam_sparse, lam_d=lam_d, lam_w=lam_w, lam_t=lam_t, c0=c0, c1=c1)



  sejwal.fit(matrixD, M_ns, vocab_pt)

  topicfile = DATA_DIR_Results + 'Provide the text file name'
  topicembeddingfile = DATA_DIR_Results + 'Provide the text file name'

  np.savetxt(topicembeddingfile, sejwal.alpha)

##########################################################################################################################

## Embedding generations for the reviews of most similar user/item


def embeddingsgeneration1():

  import preprocess1
  import preprocess
  from scipy import sparse
  import numpy as np
  import os
  import pandas as pd
  from scipy import sparse
  import CoEmbedding

  DATA_DIR_Results = 'Provide the directory name'
  DATA_DIR = 'Provide the directory name'
  doc_pt = 'Provide the directory name'
  dwid_pt = DATA_DIR+'Provide the text file name'

  dwmatrix_pt = DATA_DIR+'Provide the csv file name'
  voca_pt = DATA_DIR+'Provide the text file name'
  batch_size = 1000
  window_size = 5
  

  n_docs,w2id = preprocess1.indexFile(doc_pt,dwid_pt,dwmatrix_pt,batch_size,window_size)
  
  n_words = len(w2id)

  print('n(d)=', n_docs, 'n(w)=', n_words)

  preprocess1.write_w2id(voca_pt,w2id)

  matrixD = preprocess1.load_data(dwmatrix_pt, (n_docs, n_words))

  print matrixD

  start_idx = list(range(0, n_docs, batch_size))

  end_idx = start_idx[1:] + [n_docs]

  matrixW = sparse.csr_matrix((n_words, n_words), dtype='float32')

  for lo, hi in zip(start_idx, end_idx):
    matrixW = preprocess1._matrixw_batch(lo, hi, matrixW, n_words)
    print(float(matrixW.nnz) / np.prod(matrixW.shape))


  np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'), matrixW.data) # co-occurence matrix of each word with total no word frequency with another word
  np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'), matrixW.indices) # gives an array about the position on which we have data
  np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'), matrixW.indptr) # gives an array describes the starting postion of data in each coloumn in matrix

    
  print(matrixD.shape, matrixW.shape)

  
  tp = pd.read_csv(dwmatrix_pt)

  rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])

  matrixD = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=(n_docs, n_words))
  import main1
  matrixD = main1.tfidf(matrixD,n_docs, normalize=True)

  data = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'))
  indices = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'))
  indptr = np.load(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'))
  matrixW = sparse.csr_matrix((data, indices, indptr), shape=(n_words, n_words))

  print(matrixD.shape, matrixW.shape)
  print(float(matrixD.nnz) / np.prod(matrixD.shape))
  print(float(matrixW.nnz) / np.prod(matrixW.shape))

  count = np.asarray(matrixW.sum(axis=1)).ravel() # gives the totla sum of each row
  n_pairs = matrixW.data.sum() # total sum of all matrix values

#constructing the SPPMI matrix
  M = matrixW.copy() # copy matrix into another one.
  for i in range(n_words):                                        # to create a point wise mutual information
    lo, hi, d, idx = main1.get_row(M, i)
    M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
  

  #print(max(M.data))
  #print(M[0,0])
  
  M.data[M.data < 0] = 0
  M.eliminate_zeros()
  print(float(M.nnz) / np.prod(M.shape))

  k_ns = 1
  M_ns = M.copy()
  if k_ns > 1:
    offset = np.log(k_ns)
  else:
    offset = 0.
  
  M_ns.data -= offset
  M_ns.data[M_ns.data < 0] = 0
  M_ns.eliminate_zeros()
  print((np.absolute(M_ns).sum())/np.prod(M_ns.shape))
  vocab_pt = voca_pt
  n_embeddings = 50
  max_iter = 20
  n_jobs = 8
  c0 = 1
  c1 = 1
  K = 20
  lam_sparse_d = 1e-2
  lam_sparse = 1e-7
  lam_d = 0.5
  lam_w = 1
  lam_t = 50
  save_dir = os.path.join(DATA_DIR, 'results_parallel')
  sejwal = CoEmbedding.CoEmbedding(n_embeddings=n_embeddings, K=K, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs, 
            random_state=98765, save_params=True, save_dir=save_dir, verbose=True, 
            lam_sparse_d=lam_sparse_d, lam_sparse=lam_sparse, lam_d=lam_d, lam_w=lam_w, lam_t=lam_t, c0=c0, c1=c1)



  sejwal.fit(matrixD, M_ns, vocab_pt)

  topicfile = DATA_DIR_Results + 'Provide the text file name'
  topicembeddingfile = DATA_DIR_Results + 'Provide the text file name'

  np.savetxt(topicembeddingfile, sejwal.alpha)
  
###############################################################################################################


## Computing the rating prediction using topic embedding 


def preprocessing(request): 
  
  from nltk import word_tokenize
  from nltk.corpus import stopwords
  import nltk
  import preprocess
  from scipy import sparse
  import numpy as np
  import os
  import pandas as pd
  from scipy import sparse
  import CoEmbedding
  from collections import defaultdict
  

  stop = set(stopwords.words('english'))
  user_list = []
  itemlistu=[]
  fileitem = open('itemlist1.txt','w')
  mu = 4.02853 #(average rating value of user/item)
    
  buisness_list = ##Provide the items id
  
  for buisness in buisness_list:
    item_rating_sum = []
    for j in Yelp_NYC.objects.filter(product_id__exact=buisness):
      user_list.append(j.reviewer_id)

      item_rating_sum.append(int(j.review_ratings))


    bi = (sum(item_rating_sum)*1.0/len(item_rating_sum))

    if (mu>bi):
      bi = mu - bi
      bi = bi*(-1.0)
    if (mu<bi):
      bi=bi - mu

    for k in range(len(user_list)): 

      #import pdb;pdb.set_trace()
      user_rating_sum = []

      for ratingsum in Yelp_NYC.objects.filter(reviewer_id__exact=user_list[k]):
        user_rating_sum.append(int(ratingsum.review_ratings))

      #import pdb;pdb.set_trace()

      bu = (sum(user_rating_sum)*1.0/len(user_rating_sum))    
      
      first_user_list=[]

      for i in Yelp_NYC.objects.filter(reviewer_id__exact=user_list[k]):
        first_user_list.append(i.product_id)
      

      if (mu>bu):
        bu = mu - bu
        bu=bu*(-1.0)
      if (mu<bu):
        bu=bu-mu    


      print k
      print first_user_list
      m=k+1
      final_similarity = 0
      denominator_sim = 0
      while m<len(user_list):
        second_user_list=[]
        review_rating = []
        for r in Yelp_NYC.objects.filter(reviewer_id__exact=user_list[m]):
          second_user_list.append(r.product_id)
        print m
        print second_user_list
        
        common_item=(list(set(first_user_list) & set(second_user_list)))
        f = open('Provide the file name','w')
        f1 = open('Provide the file name','w')
        d = defaultdict(list)
        d1 = defaultdict(list)
        if 'product_id__exact' in common_item:  ## write the name of the product id here(buisness_id)
          common_item.remove('product_id')
        
        for s in common_item:


          
          for y in Yelp_NYC.objects.filter(reviewer_id__exact=user_list[k]):
            if y.product_id==s:
              first_reiew = y.review.replace("\n","")           
              d[y.product_id].append(y.review_classification)
              d[y.product_id].append(first_reiew)
            
              #f.write(y.user_reviews.replace("\n",""))
              print'======================'
                
          for z in Yelp_NYC.objects.filter(reviewer_id__exact=user_list[m]):
             
            if z.product_id==s:
              second_review = z.review.replace("\n","")

              d1[z.product_id].append(z.review_classification)
              d1[z.product_id].append(second_review)
              d1[z.product_id].append(z.review_ratings)
              
              #f1.write(z.user_reviews.replace("\n",""))
              print z.review_classification
              print '+++++++++++++++++++++'
              
        for key in range(len(common_item)):
          if (d.keys()[key] == d1.keys()[key]):
            if (d[d.keys()[key]][0]==d1[d1.keys()[key]][0]):
              
              #print d[d.keys()[key]][1]
              if d[d.keys()[key]][1] is not None:
                f.write(d[d.keys()[key]][1].encode('utf-8'))
              f.write('\n')
              print 'vineet============================================================='
              #print d1[d1.keys()[key]][1]
              if d1[d1.keys()[key]][1] is not None:
                f1.write(d1[d1.keys()[key]][1].encode('utf-8'))
              f1.write('\n')  
              print 'sejwal+++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
              review_rating.append(d1[d1.keys()[key]][2])
              
        f1.close()    
        f.close()  
        
        
        if os.path.getsize("doc.txt") > 0:
          fdoc = documentfilter()
          fqq =open('newfilterdoc.txt','r')

          docembedding1 = embeddingsgeneration()  


        else:
          print 'sejwal111111111111111111111111111111111111111'
        
        if os.path.getsize("doc1.txt") > 0:
          fdoc1 = documentfilter1()
          fqq1 = open('newfilterdoc1.txt','r')

          docembedding2 = embeddingsgeneration1()

          common_item_embed = []
          common_item_embed1 = []


          file1 = open("Provide the location of the embedding text file","r") 

          for kq in file1:
            q=  kq.split()
            common_item_embed.append(q)

          file2 = open("Provide the location of the embedding text file","r") 

          for w in file2:
            r=  w.split()
            common_item_embed1.append(r)

            


          if 'nan' in open("Provide the location of the embedding text file","r").read():
            print 'Similarity remain same as previous one'
            final_similarity = final_similarity
            denominator_sim = denominator_sim


          else:

            from scipy import spatial
            
            embeddings_sim_sum = 0
            
            for ii in range(len(common_item_embed)):

              a = map(float, common_item_embed[ii])
              b = map(float, common_item_embed1[ii])

              result = 1 - spatial.distance.cosine(a, b)
              embeddings_sim_sum = embeddings_sim_sum + result

            if (embeddings_sim_sum<0):
            
              embeddings_sim_sum = embeddings_sim_sum * (-1.0)    

            embedding_similarity = ((embeddings_sim_sum *1.0)/len(common_item_embed))   
              
            print 'final similarity is',((embeddings_sim_sum *1.0)/len(common_item_embed)) 

            denominator_sim = (len(review_rating)*embedding_similarity) + denominator_sim

            sim_sum = 0

            for rating in review_rating:

              sim_sum = ((int(rating) - (mu+bu+bi))*embedding_similarity)+sim_sum 


            final_similarity = final_similarity+sim_sum

            print 'final value is',(final_similarity*1.0)/denominator_sim
         
        
        
        m = m+1  
      #import pdb;pdb.set_trace()
      try:    
        final_prediction = mu+bi+bu + ((final_similarity*1.0)/denominator_sim)
      except ZeroDivisionError:
          final_prediction = mu+bi+bu
          

      fileresult = open('provide the text file name','a')

      for ij in Yelp_NYC.objects.filter(reviewer_id__exact=user_list[k]):
        if int(ij.product_id) == buisness:
          
          fileresult.write(ij.reviewer_id)
          fileresult.write('\t')
          fileresult.write(ij.review_ratings)
          fileresult.write('\t')
      
      fileresult.write(str(final_prediction))     
      fileresult.write('\n')  
      fileresult.close()  
    #import pdb;pdb.set_trace()

  
  

