# coding: utf-8


#
# hw6 problem 2 (also a lab problem, _if_ you join two labs!)
#

## Problem 2: Analogies!


# A helper function - are all words in the model?
#
def all_words_in_model( wordlist, model ):
    """ returns True if all w in wordlist are in model
        and False otherwise
    """
    for w in wordlist:
        if w not in model:
            return False
    return True


# Here's a demonstration of the fundamental capability of word2vec on which
#   you'll be building:  most_similar

def test_most_similar(model):
    """ example of most_similar """
    print("Testing most_similar on the king - man + woman example...")
    LoM = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
    # note that topn will be 100 below in check_analogy...
    return LoM


#
#
# Start of functions to write + test...
#
#


#
# Write your generate_analogy function
#
def generate_analogy(word1, word2, word3, model):
    """ takes in three words, and gives the best analogy to word 3
    """
    LoM = model.most_similar(positive=[word3, word2], negative=[word1], topn=10)
    return LoM[0][0]


#
# Write your check_analogy function
#
def check_analogy(word1, word2, word3, word4, model):
    """ checks the words (takes in word 1 : word 2, word 3 : ?), and checks if it's elsewhere
    """
    LoM = model.most_similar(positive=[word3, word2], negative=[word1], topn=10)

    print(LoM)
    for x in range(len(LoM)):
        if LoM[x][0] == word4:
            return 100-x
    return 0


#
# Results and commentary...
#

#
# (1) Write generate_analogy and try it out on several examples of your own
#     choosing (be sure that all of the words are in the model --
#     use the all_words_in_model function to help here)
#
# (2) Report two analogies that you create (other than the ones we looked at in class)
#     that _do_ work reaonably well and report on two that _don't_ work well
#     Finding ones that _do_ work well is more difficult! Maybe in 2025, it'll be the opposite (?)

# WORKS:
# In [25]: generate_analogy('female', 'woman', 'male', m)
# Out[25]: 'man'

# In [29]: generate_analogy('girl', 'boy', 'woman', m)
# Out[29]: 'man'

# DOESN'T
# In [26]: generate_analogy('teacher', 'student', 'boss', m)
# Out[26]: 'bosses'

# In [28]: generate_analogy('short', 'tall', 'small', m)
# Out[28]: 'tiny'


#
#
# (3) Write check_analogy that should return a "score" on how well word2vec_model
#     does at solving the analogy given (for word4)
#     + it should determine where word4 appears in the top 100 (use topn=100) most-similar words
#     + if it _doens't_ appear in the top-100, it should give a score of 0
#     + if it _does_ appear, it should give a score between 1 and 100: the distance from the
#       _far_ end of the list. Thus, a score of 100 means a perfect score. A score of 1 means that
#       word4 was the 100th in the list (index 99)
#     + Try it out:   check_analogy( "man", "king", "woman", "queen", m ) -> 100
#                     check_analogy( "woman", "man", "bicycle", "fish", m ) -> 0
#                     check_analogy( "woman", "man", "bicycle", "pedestrian", m ) -> 96





#
#
# (4) Create at least five analogies that perform at varying levels of "goodness" based on the
#     check_analogy scoring criterion -- share those (and any additional analysis) with us here!

#In [30]: check_analogy('teacher', 'student', 'boss', 'employee', m)
#[('bosses', 0.42561209201812744), ('bossman', 0.3768966495990753), ('chairman', 0.3574698865413666), ('exec', 0.3565175533294678), ('president', 0.3537697196006775), ('mates', 0.34461379051208496), ('roommate', 0.33668577671051025), ('Boss', 0.3298180103302002), ('mate', 0.3222281038761139), #('Student', 0.31689077615737915)]
#Out[30]: 0

#In [31]: check_analogy('short', 'tall', 'small', 'large', m)
#[('tiny', 0.5305808782577515), ('large', 0.48962604999542236), ('smallish', 0.47144025564193726), ('taller', 0.46704137325286865), ('diminutive', 0.4579048752784729), ('hulking', 0.438798725605011), ('gigantic', 0.4352542757987976), ('towering', 0.42367979884147644), ('sized', 0.4213898777961731), #('smaller', 0.4172428250312805)]
#Out[31]: 99

#In [35]: check_analogy('king', 'queen', 'husband', 'wife', m)
#[('daughter', 0.6421603560447693), ('wife', 0.6357147693634033), ('mother', 0.624269962310791), ('fiance', 0.6129336357116699), ('niece', 0.563124418258667), ('hubby', 0.5612773895263672), ('daughters', 0.5558146238327026), ('her', 0.5552783012390137), ('granddaughter', 0.5516073703765869), ##('exhusband', 0.5445963144302368)]
#Out[35]: 99

#In [36]: check_analogy('big', 'small', 'tall', 'short', m)
#[('Tall', 0.4728952646255493), ('diminutive', 0.4572795331478119), ('slender', 0.4478764533996582), ('taller', 0.4448097348213196), ('tiny', 0.4446330666542053), ('diameter', 0.44094476103782654), ('cylindrical', 0.4289061427116394), ('wiry', 0.42346930503845215), ('stumpy', 0.41325151920318604), ('unimposing', 0.4103904962539673)]
#Out[36]: 0

#In [37]: check_analogy('big', 'small', 'long', 'short', m)
#[('short', 0.42859798669815063), ('longer', 0.37502560019493103), ('tiny', 0.37263500690460205), ('Long', 0.3677181601524353), ('medium', 0.36243483424186707), ('narrow', 0.36240026354789734), ('lengthy', 0.357666552066803), ('brief', 0.3181127607822418), ('relatively', 0.31578171253204346), ('continuous', 0.3096880316734314)]
#Out[37]: 100