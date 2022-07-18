# translate word into id in documents
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
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import itertools
import os

#w2id = {}
DATA_DIR = 'E:/results/'

def indexFile(pt, res_pt, matrix_pt, batch_size, window_size):
    w2id = {}
    print('index file: ', pt)
    wf = open(res_pt, 'w')
    wf_matrix = open(matrix_pt, 'w')
    wf_matrix.writelines('doc_id,word_id')
    docid = 0
    saveid = 0
    rows = []
    cols = []
    for l in open(pt):
        ws = l.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id) # creating a dictionary with words and indexing
        

        wids = [w2id[w] for w in ws]   # creating doc_id  
        #print wids
        #print map(str, wids)
        #print 'm'.join(map(str, wids))
        #print 'vineet'
        wf.writelines(' '.join(map(str, wids))) # join function join words id with blank (' ') to make doc id
        wf.writelines('\n')
        tem_sep = '\n' + str(docid) +',' # creating a temp variable with docid intial value (doc id will run one time i.e initaite)   

        wf_matrix.writelines('\n'+str(docid)+','+tem_sep.join(map(str,wids))) # creating a matrix with first part to write in doc id second part in word id
        docid = docid + 1
        #print '\n'+str(docid)+','+tem_sep.join(map(str,wids))
        #print "fazil" 
        #print (tem_sep.join(map(str,wids)))

        for ind_focus, wid_focus in enumerate(wids): # enumerate (wids) will give (0,docid), (1,docid),(2,docid),(3,docid),(4,docid)...
            ind_lo = max(0, ind_focus-window_size)
            ind_hi = min(len(wids), ind_focus+window_size+1) # len(wids) gives len of each document, total no of words
            '''
            for wid_con in wids[ind_lo: ind_hi]: # step 42 to 44 are used for determining window size +-5
                rows.append(wid_focus)
                cols.append(wid_con)
            '''
           # print ind_focus, "=******=", wid_focus
            #print ind_lo, "===", ind_hi
            for ind_c in range(ind_lo, ind_hi):
                #print ind_c, "====", ind_focus
                if ind_c == ind_focus:
                    #print ind_c, "==ind_focus==", ind_focus
                    continue
                '''diagonals are zeros or not'''
                #print  wid_focus, "+++++", wids[ind_c]
                if wid_focus == wids[ind_c]:
                   # print  wid_focus, "==wids[ind_c]==", wids[ind_c]
                    continue
                rows.append(wid_focus)
                cols.append(wids[ind_c])
                #print "rows==", rows
               # print "cols==", cols

     # steps 42 to 65 are used to generate rows and cols based on window size embeedingsE
        if docid%batch_size == 0 and docid != 0:
            np.save(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (saveid, docid)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
            saveid = saveid + batch_size

            print('%dth doc, %dth doc' % (saveid, docid))
            rows = []
            cols = []
    np.save(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (saveid, docid)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))      
    
    wf.close()
    wf_matrix.close()

    print('write file: ', res_pt)
    print('write file: ', matrix_pt)
    #import pdb;pdb.set_trace()
    return docid, (w2id)
    

def write_w2id(res_pt,w2id):
    print('write:', res_pt)
    wf = open(res_pt, 'w')
    for w, wid in sorted(w2id.items(), key=lambda d:d[1]):
        #print '%d\t%s' % (wid, w)
        wf.writelines('%d\t%s' % (wid, w))
        wf.writelines('\n')
        
    wf.close()
    
def load_data(csv_file, shape):
    
    print('loading data')
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])

    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape) # a row sparse matrix is generated. 
   
    return data

def _coord_batch(lo, hi, train_data):
    rows = []
    cols = []
    for u in range(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
        if u%1000 == 0:
            print('%dth doc' % u)
    np.save(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))

def _matrixw_batch(lo, hi, matW,n_words):
    
    coords = np.load(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi))) #matrix generated from step 42 to 65   
    rows = coords[:, 0]
    cols = coords[:, 1]    
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_words, n_words), dtype='float32').tocsr() #to sum duplicate matrix
    matW = matW + tmp
    
    print("User %d to %d finished" % (lo, hi))
    sys.stdout.flush()
    
    return matW
   