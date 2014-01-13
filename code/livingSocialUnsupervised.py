import pandas as pd
import pickle,nltk,string,re,json,pprint,networkx
from matplotlib.pyplot import *
from gensim import models
from gensim import corpora
from sklearn.cluster import KMeans

###pattern.en.lemma("")
def tokensToVW(tokens):
    #of the form ID |F_name1 <text string> |F_name2 <text string> ...
    return -1

######################################################
def nltkParseText(text,returnTokens=True):

    #nltk filtering
    stemmer = nltk.PorterStemmer()

    fblock = nltk.clean_html(text)
    fblock = string.lower(fblock)

    fblock = ''.join(re.findall('[a-z\s]',fblock)) #get rid of all special characters except spaces
    tokens = nltk.word_tokenize(fblock) #tokenize by word
    tokensClean = [stemmer.stem(w) for w in tokens] #stem the words

    if returnTokens:
        return tokensClean
    else:
        return ' '.join(tokensClean)
######################################################
def gensimParseText(text,globalDict=None):
    #gensim filtering
    stoplist = set('for a of the and to in'.split())
    dictionary = corpora.Dictionary(line.lower().split() for line in text)
    #dictionary = corpora.Dictionary(text)
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
    dictionary.compactify() # remove gaps in id sequence after words that were removed

    if globalDict == None:
        return dictionary
    else:
        dict2_to_dict1 = globalDict.merge_with(dictionary)
        return dict2_to_dict1,dictionary

######################################################
def parseJSON(jsonText):
    features = ['country_code','campaign_name',\
    'description','long_title','deal_concept']

    #first pass over every text doc
    myText = ''
    for doc in jsonText:
        for f in features:
            myText+=doc[f]
    globalDict=gensimParseText(nltkParseText(myText))
    #print globalDict.token2id

    tfidf=models.TfidfModel(dictionary=globalDict)
    print tfidf[globalDict]
    print '------------------'

    #corp is a dict
    #corp doc2bow('text'.split()

    #second pass over individual text docs
    #return a mapping from individual doc's tokens to global doc tokens
    fv = dict()
    for i,doc in enumerate(jsonText):
        myText = ''
        for f in features:
            myText+=doc[f]
        fv[i]=globalDict.doc2bow(nltkParseText(myText))

    return fv #return feature vector for each document
######################################################
def cosineSimilarity(x,y):
    if x.sum()==0 or y.sum()==0:
        return 0
    else:
        return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

######################################################
def clusterFV(featureDict):
    #TODO: map feature list to a 2D numpy array

    print featureDict
    X = np.zeros([3000,len(featureDict.keys())])
    for key in featureDict:
        for value in featureDict[key]:
            X[value[0]][key]=value[1]

    csVec = np.array([])
    ivec = np.array([])
    jvec = np.array([])
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            csVec = np.append(csVec,cosineSimilarity(X[:,i],X[:,j]))
            if csVec[-1]>0.5 and csVec[-1]<.95:
                ivec = np.append(ivec,i)
                jvec = np.append(jvec,j)
    pickle.dump((ivec,jvec,csVec),open('../data/csVec.pkl','wb'))
    '''
    nclusters = [2,4,8,16,32]
    for ncluster in nclusters:
        predCluster = KMeans.fit_predict(featureVector)
    '''
    return ivec,jvec

######################################################
#not used.
def readJSON(fname):

    #readMeta('../data/json_output.txt')
    with open(fname) as f:
        for line in f.readlines():
            split = re.split('\[([^]]+)\]',line)
            out=''
            tag = list()
            for i in range(len(split)):

                if len(split[i])>=50:
                    out = ''.join([out,split[i]])
                elif len(split[i])>10 and len(split[i])<50:
                    tag.append(split[i])

            return tag,out

######################################################
if __name__=='__main__':

    #mytags,myjson = readJSON('../data/in.json')

    with open('../data/in.json','r') as f:
        for lines in f.readlines():
            print '____________'
            data = json.loads(lines)

            data_parsed = parseJSON(data)

            #print data_parsed
            #ivec,jvec = clusterFV(data_parsed)

            ivec,jvec,csvec = pickle.load(open('../data/csvec.pkl','rb'))
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
