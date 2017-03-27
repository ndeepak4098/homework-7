# coding: utf-8

#
# Lab problem #1, part 1   Make sure everything's installed and working!
#

## Before you go any further, make sure that you can load the word2vec model.
## If the import statement fails, make sure that you have run 
## conda install gensim
## in your terminal.

#
# run with m = read_word2vec_model()  or model = ...
#
def read_word2vec_model():  
    """ a function that reads a word2vec model from the file
        "word2vec_model.txt" and returns a model object that
        we will usually name m or model...
    """
    file_name = "word2vec_model.txt"
    from gensim.models import KeyedVectors
    m = KeyedVectors.load_word2vec_format(file_name, binary=False)
    print("The model built is", m)
    ## The above line should print
    ## Word2Vec(vocab=43981, size=300, alpha=0.025)
    ## which tells us that the model represents 43981 words with 300-dimensional vectors
    ## The "alpha" is a model-building parameter called the "learning rate."
    ##   Once the model is built, it can't be changed without rebuilding it; we'll leave it.  
    return m


def most_similar_example(m):
    """ showing off most_similar """
    print("Testing most_similar on the king - man + woman example...")
    LoM = m.most_similar(positive=['woman', 'king'], negative=['man'])
    return LoM


def doesnt_match_example(m):
    """ showing off doesnt_match """
    LoW = "breakfast cereal dinner lunch".split()
    print("Testing doesnt_match on the example with LoW =")
    print(LoW)
    nonmatcher = m.doesnt_match(LoW)
    return nonmatcher



#
# Practice using textblob to tokenize some text
#
import nltk
import textblob

#
# from this week's problem 0 (http://nlp.stanford.edu/sentiment/index.html)
#
example_text = """
The underlying technology of this demo is based on a new type of Recursive Neural Network that builds on top of grammatical structures. You can also browse the Stanford Sentiment Treebank, the dataset on which this model was trained. The model and dataset are described in an upcoming EMNLP paper. Of course, no model is perfect. You can help the model learn even more by labeling sentences we think would help the model or those you try in the live demo.
"""

def textblob_examples(example_text=example_text):
    """ showing off the textblob and nltk libraries,
        first, to check if they work at all...
    """
    # tokenize with NLTK
    print("Here is the tokenized list-of-words from example_text:")
    LoW = nltk.word_tokenize(example_text.lower())
    print(LoW)
    print()

    print("And a list-of-sentences from example_text:")
    LoS = nltk.sent_tokenize(example_text.lower())
    print(LoS)
    print()

    # tokenize with textblob - first create a blob...
    blob = textblob.TextBlob( example_text )
    print("Tokenizing example with textblob:")
    print("Words:")
    print( blob.words )
    print("Sentences:")
    print( blob.sentences )


#
# Lab problem #1, part 2   Work through the TextBlob QuickStart Tutorial
#
# here (actually, inside OR outside the above function), try out the examples from the 
# TextBlob QuickStart Tutorial, at https://textblob.readthedocs.io/en/dev/quickstart.html
#
# OR, feel free to do this at the Python prompt, instead!
#

# 
blob = textblob.TextBlob( example_text )
blob.tags

#
# What are those part-of-speech tags?  They are here:
#    http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# 



#
# now, run with this week's data (and your own wordsets)
#

#
# run this, for example, with 
#     visualize_wordvecs(["breakfast", "lunch", "cereal", "dinner"], m)
#
def visualize_wordvecs(wordlist, model):
    """ example of finding an outlier with word2vec and graphically """
    # 
    # Are all of the works in the model?
    #
    for w in wordlist:
        if w not in model:
            print("Aargh - the model does not contain", w)
            print("Stopping...")
            return
    # 
    # First, find word2vec's outlier:
    #
    outlier = model.doesnt_match(wordlist)
    print("{0} is not like the others.".format(outlier))

    #
    # Next, we use PCA, Principal Components Analysis, to reduce dimensionality
    # and create a scatterplot of the words...
    #
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy

    pca = PCA(n_components=2)   # 2 dimensions
    pca_model = pca.fit(model.syn0)  # all 43,981 words with 300 numbers each!

    LoM = [model[w] for w in wordlist]   # list of models for each word w
    word_vectors = numpy.vstack(LoM)     # vstack creates a vertical column from a list
    transformed_words = pca_model.transform(word_vectors)  # transform to our 2d space

    # scatterplot
    plt.scatter(transformed_words[:,0],transformed_words[:,1])
    
    # This is matplotlib's code for _annotating_ graphs (yay!)
    for i, word in enumerate(wordlist):
        plt.annotate(word, (transformed_words[i,0], transformed_words[i,1]), size='large')
        # it's possible to be more sophisticated, but this is ok for now

    plt.show()
    return


#
# Your tasks for the third part of this hw6pr1 (lab problem)
#
#   (1) Find two lists of four-or-more words (all in the model) where visualize_wordvecs
#       does a _good_ job of identifying an outlier - note them and the results here:


    # 1. visualize_wordvecs(["green", "yellow", "orange", "apple"], m)
    #apple is not like the others.

    # 2. visualize_wordvecs(["jeans", "shirt", "pants", "backpack"], m)
    #backpack is not like the others.

#
#   (2) Find two lists of four-or-more words (all in the model) where it's possible
#       to see that visualize_wordvecs has _missed_ the outlier (in some sense - you choose)
#       Note these and the results here:

    # 1. visualize_wordvecs(["female", "male", "girl", "woman"], m)
    #girl is not like the others.

    # 2. visualize_wordvecs(["phone", "car", "computer", "television", "toy"], m)
    #television is not like the others.

#
#   (3) Include your four plots as screenshots or saved as images using matplotlib
#       please save them as outlier1.png outlier2.png outlier3.png and outlier4.png
#









#
#   (Extra) Change the PCA dimensionality-reduction to 3d using the iris example, below, as
#       a guide, and then find a set of words in which there are two outliers -- one
#       in each of two different directions !
#
#   (5) Again, share the results, including a screenshot (at a strategically chosen pose)
#




#
# connect to the last two weeks' of data (the iris dataset)
#


def iris_pca():
    """ example of projecting the 4d iris data onto the "best" 3d axes
        that is, the axes along which the data spreads the most...
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn.decomposition import PCA
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf() ; plt.cla()  # clear figure and clear axes
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)  # create axes
    
    # reduce from 4 dimensions to 3
    pca = PCA(n_components=3)
    pca.fit(X)
    # X = iris.data[:,1:4]  # don't use PCA
    X = pca.transform(X)    # do use PCA
    
    # Add labels
    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        xpos = X[y == label, 0].mean()
        ypos = X[y == label, 1].mean() + 1.5
        zpos = X[y == label, 2].mean()
        # plot label
        ax.text3D(xpos, ypos, zpos, name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

    # Now, plot the points themselves
    y = np.choose(y, [1, 2, 0]).astype(np.float)  # this chooses a color for each flower type
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)
    # no labels - qualitative comparison
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

