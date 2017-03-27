# coding: utf-8


#
# hw6 problem 3
#

## Problem 3: Paraphrasing!

import textblob
from textblob import Word

# A starter function that substitutes each word with it's top match
#   in word2vec.  Your task: improve this with POS tagging, lemmatizing, 
#   and/or at least three other ideas of your own (more details below)
#
def paraphrase_sentence( sentence, model ):
    """
    This will take in a sentence, and create a new sentence with similar words that don't start with the same letter, or contain e
    """
    # 3 implementations:
    # if the first letter is equal to the first letter of w, move to the next one.
    # don't use the letter e at all
    # spelling corrections

    blob = textblob.TextBlob( sentence )
    print("The sentence's words are")
    LoW = blob.words
    print(LoW)

    NewLoW = ""
    for w in LoW:
        
        if w not in model:
            NewLoW += w + " "
        else:
            
            w_alternatives = model.most_similar(positive=[w], topn=100)

            counter = 0

            for i in range(len(w_alternatives)):
                
                first_alternative, first_alternative_score = w_alternatives[i]  # initial one!

                if (first_alternative[0] != w[0]):
                    if 'e' not in first_alternative:
                        break
                else:
                    counter += 1
                    
            if counter == len(w_alternatives):
                first_alternative, first_alternative_score = w_alternatives[0]
            else:   
                NewLoW += first_alternative + " "
    
    # you should change this so that it returns a new string (a new sentence),
    # NOT just print a list of words (that's what's provided in this starter code)
   
    NewLoW = NewLoW[:-1]
    
    NewLoW = textblob.TextBlob(NewLoW)
    NewLoW = NewLoW.correct()
    NewLoW = NewLoW.words
    
    NewSentence = ""
    
    for x in NewLoW:
        NewSentence += x + " "
    
    NewSentence = NewSentence[:-1]   
    
    return NewSentence


# 
# Once the above function is more sophisticated (it certainly does _not_ need to be
#   perfect -- that would be impossible...), then write a file-paraphrasing function:
#
def paraphrase_file(filename, model):
    """
    This will take in a file, and return a new file with the equivalent sentences in paraphrased sentences
    """
    n = open(filename)
    data= n.readlines()
    
    for x in data:
        f = open("test_paraphrased.txt", "w")
        f.write(paraphrase_sentence(x, model))
    
    f.close()
    n.close()

#
# Results and commentary...
#


# (1) Try paraphrase_sentence as it stands (it's quite bad...)  E.g.,
#         Try:    paraphrase_sentence("Don't stop thinking about tomorrow!", m)
#         Result: ['Did', "n't", 'stopped', 'Thinking', 'just', 'tonight']

#     First, change this so that it returns (not prints) a string (the paraphrased sentence),
#         rather than the starter code it currently has (it prints a list) Thus, after the change:

#         Try:    paraphrase_sentence("Don't stop thinking about tomorrow!", m)
#         Result: "Did n't stopped Thinking just tonight"  (as a return value)

#     But paraphrase_sentence is bad, in part, because words are close to variants of themselves, e.g.,
#         + stop is close to stopped
#         + thinking is close to thinking





# (2) Your task is to add at least three things that improve this performance (though it
#     will necessarily still be far from perfect!) Choose at least one of these two ideas to implement:

#     #1:  Use lemmatize to check if two words have the same stem/root - and _don't_ use that one!
#             + Instead, go _further_ into the similarity list (past the initial entry!)
#     #2:  Use part-of-speech tagging to ensure that two words can be the same part of speech

#     Then, choose two more ideas that use NLTK, TextBlob, or Python strings -- either to guard against
#     bad substitutions OR to create specific substitutions you'd like, e.g., just some ideas:
#        + the replacement word can't have the same first letter as the original
#        + the replacement word is as long as possible (up to some score cutoff)
#        + the replacement word is as _short_ as possible (again, up to some score cutoff...)
#        + replace things with their antonyms some or all of the time
#        + use the spelling correction or translation capabilities of TextBlob in some cool way
#        + use as many words as possible with the letter 'z' in them!
#        + don't use the letter 'e' at all...
#     Or any others you might like!





# (3) Share at least 4 examples of input/output sentence pairs that your paraphraser creates
#        + include at least one "very successful" one and at least one "very unsuccessful" ones

# Very Successful:
# In [85]: paraphrase_sentence("It is cold", m)
# Out[85]: 'That was frigid'

# Very Unsucessful:
# paraphrase_sentence("my english is not good", m)
# Out[80]: 'By english was do bad'

# Average:
# paraphrase_sentence("I love you", m)
# Out[78]: 'my passion You'

# Average:
# paraphrase_sentence("Computer science is awesome", m)
# Out[79]: 'Computer biology was fantastic'



# (4) Create a function paraphrase_file that opens a plain-text file, reads its contents,
#     tokenizes it into sentences, paraphrases all of the sentences, and writes out a new file
#     containing the full, paraphrased contents with the word paraphrased in its name, e.g.,
#        + paraphrase_file( "test.txt", model )
#             should write out a file names "test_paraphrased.txt"  with paraphrased contents...
#        + include an example file, both its input and output -- and make a comment on what you
#             chose and how it did! 

# I chose to write the first paragraph of Abraham Lincoln's Gettysburgh address. It did okay, but it literally translated every word, one by one, (which we can see the code does)
# which made it very hard to understand. Also, since some words were capitalized randomly, which made it even harder to understand the already broken down englush. However, when I 
# took a second look at each word, I saw that it didn't perform as badly as I thought, as the code did find a similar word that may have worked in some circumstances. Obviously, thought,
# it wasn't perfect.

# I inputed:
# Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.

# It outputed:
# Two play and four months last Our husbands put back on last Africa a changing country built In tyranny and focus to in notion it All boys is built Equal











# (Optional EC) For extra-credit (up to +5 pts or more)
#        + [+2] write a function that takes in a sentence, converts it (by calling the function above) and
#          then compares the sentiment score (the polarity and/or subjectivity) before and after
#          the paraphrasing
#        + [+3 or more beyond this] create another function that tries to create the most-positive or
#          most-negative or most-subjective or least-subjective -- be sure to describe what your
#          function does and share a couple of examples of its input/output...
