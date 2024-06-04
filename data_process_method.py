
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def data_process(t):
    t=t.lower()
    t=nltk.word_tokenize(t)
    l=[]
    for i in t:
        if i.isalnum():
            l.append(i)
    t=list(l)
    l.clear()
    for i in t:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)
    t=list(l)
    l.clear()
    for i in t:
        l.append(ps.stem(i))
    
    return " ".join(l)


