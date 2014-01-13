
import pandas as pd
import pickle,string,re,json,pprint,networkx,nltk
from matplotlib.pyplot import *
from gensim import models,corpora,similarities,matutils
from sklearn.cluster import KMeans
from lshash import LSHash

class MyCorpus:

    def __init__(self,jsonf):

        self.fname = 'myArray.txt'

        features = ['country_code','campaign_name',\
        'description','long_title','deal_concept']

        jsonList = json.load(open(jsonf,'r'))

        myList = []
        myStr = ''
        for jsonDoc in jsonList:
            myStr=''
            for f in features:
                myStr+=jsonDoc[f]

            line = re.sub('\<[^>]*\>',' ',myStr).lower()
            line = re.sub('[^a-z|^0-9|^ ]','',line).lower()
            myList.append(line)

        fout = open(self.fname,'w')
        for v in myList:
            fout.write(v+'\n')
        fout.close()

    def buildDictionary(self):
        self.dictionary = corpora.Dictionary(line.split() for line in open(self.fname,'r'))

        stoplist = set('for a of the and to in by an'.split())
        stopIds = [self.dictionary.token2id[stopword] for stopword in stoplist\
            if stopword in self.dictionary.token2id]
        onceIds = [tokenid for tokenid, docfreq in self.dictionary.dfs.iteritems()\
                   if docfreq == 1]
        self.dictionary.filter_tokens(stopIds+onceIds)
        self.dictionary.compactify()

    def __iter__(self):
        for line in open(self.fname,'r'):
            yield self.dictionary.doc2bow(line.split())





######################################################
def cosineSimilarity(x,y):
    if x.sum()==0 or y.sum()==0:
        return 0
    else:
        return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

######################################################
if __name__=='__main__':
    data = json.load(open('in.json','r'))
    myCorpus = MyCorpus('in.json')
    myCorpus.buildDictionary()

    corpus = list()
    [corpus.append(line) for line in myCorpus]

    #map bag-of-words to term freq inverse doc freq {0,1}=>[0,1]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    #map tfidf to latent semantic analysis -  reduce dimensionality
    lsi = models.LsiModel(corpus_tfidf,id2word=myCorpus.dictionary,num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]

    mtx= matutils.corpus2dense(corpus_lsi,num_terms=300)

    '''
    lsh = LSHash(6,300)
    for i in range(mtx.shape[1]):
        lsh.index(mtx[:,i])

    for i in range(mtx.shape[1]):
        q= lsh.query(mtx[:,i])

        if len(q)>1:
            print len(q)
            for qi in q:
                print mtx.T==np.array([qi[0],np.newaxis])
                print data[np.where(np.all(mtx==np.array(qi[0]),axis=0))[0]]['long_title']
            print '======'
    '''
    ivec = np.array([])
    jvec = np.array([])
    cssVec = np.array([])
    for i in range(mtx.shape[1]):
        for j in range(i+1,mtx.shape[1]):
            css= cosineSimilarity(mtx[:,i],mtx[:,j])
            cssVec = np.append(cssVec,css)
            if css>.25 and css<.8:
                ivec = np.append(ivec,i)
                jvec = np.append(jvec,j)

    plot(np.sort(cssVec),'*r')
    show()

    figure()
    G = networkx.Graph()
    for idx,i in enumerate(ivec):
        ititle= data[int(i)]['long_title']
        jtitle=data[int(jvec[idx])]['long_title']
        print ititle
        print jtitle
        print '======='
        G.add_edge(ititle,jtitle)
    networkx.draw(G)
    show()

