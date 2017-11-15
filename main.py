
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import sys
import io

import gensim
from gensim import corpora


reload(sys)
sys.setdefaultencoding('utf8')


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def printInfo(corpus): # prints general information about a corpus of text (list of strings)
    print("*********** GENERAL INFO ON CORPUS **********")
    lengths = [len(i) for i in corpus]
    print "Number of items in the corpus: " + str(len(corpus))
    print "Avg length of text in the corpus: " + str(float(sum(lengths)) / len(lengths))
    docs = [doc.split() for doc in corpus]
    print "Total vocabulary of the corpus: " + str(len(corpora.Dictionary(docs)))


# compile documents
doc_fake = []
doc_real = []


print(os.getcwd())

directory = "../../temp/fake-real-news/binary/fakeonly"
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file = os.path.join(directory, filename)
        file_text = io.open(file,encoding = "ISO-8859-1").read()
        #print(file)
        doc_fake.append(file_text)
        continue
    else:
        continue

print(doc_fake[1])
printInfo(doc_fake)

directory = "../../temp/fake-real-news/binary/realonly"
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file = os.path.join(directory, filename)
        file_text = io.open(file,encoding = "ISO-8859-1").read()
        #print(file)
        doc_real.append(file_text)
        continue
    else:
        continue

print(doc_real[1])
printInfo(doc_real)


doc_complete = doc_fake + doc_real

# clean documents
doc_clean = [clean(doc).split() for doc in doc_complete]      



# Creating the term dictionary of our courpus, where every unique term is assigned an index.
print("Dictionary preparation:")
dictionary = corpora.Dictionary(doc_clean)
print(len(dictionary))
dictionary.filter_n_most_frequent(100)
print(len(dictionary))
dictionary.filter_extremes(no_below=5, no_above=0.5)
print(len(dictionary))
print("Dictionary finalized.")



# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print(len(doc_term_matrix))


# Creating the object for LDA model using gensim library
print("Creating Lda...")
Lda = gensim.models.ldamodel.LdaModel


# Running and Trainign LDA model on the document term matrix.
print("Building model...")
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=5, random_state=1)

# Show topics with some most important words
print(ldamodel.print_topics(num_topics=10, num_words=3))



# Working on a new corpus
# clean documents
fake_clean = [clean(doc).split() for doc in doc_fake]
corpus = [dictionary.doc2bow(doc) for doc in fake_clean]
print(len(corpus))
bow = corpus[1]
print(doc_fake[1])
print(ldamodel.get_document_topics(bow,minimum_probability=0.1))

real_clean = [clean(doc).split() for doc in doc_real]
corpus = [dictionary.doc2bow(doc) for doc in real_clean]
print(len(corpus))
bow = corpus[6]
print(doc_fake[6])
print(ldamodel.get_document_topics(bow,minimum_probability=0.1))


#visulalize topics by wordcloud
from wordcloud import WordCloud
from pylab import *
plt.figure(figsize=(30, ldamodel.num_topics))
subplots_adjust(hspace=0.1, wspace=0.1, top=0.1)
plt.axis("off")
for t in range(ldamodel.num_topics):
    ax1 = subplot((ldamodel.num_topics/5 +1), 5, t+1)
    ax1.imshow(WordCloud(background_color="white").fit_words(ldamodel.show_topic(t, 10)))
    ax1.set_title("Topic #" + str(t))

plt.savefig('test_all.pdf', format='pdf')

