import torch


test_sentence = """n-gram models are widely used in statistical natural 
language processing . In speech recognition , phonemes and sequences of 
phonemes are modeled using a n-gram distribution . For parsing , words 
are modeled such that each n-gram is composed of n words . For language 
identification , sequences of characters / graphemes ( letters of the 
alphabet ) are modeled for different languages For sequences of characters , 
the 3-grams ( sometimes referred to as " trigrams " ) that can be 
generated from " good morning " are " goo " , " ood " , " od " , " dm ", 
" mo " , " mor " and so forth , counting the space character as a gram 
( sometimes the beginning and end of a text are modeled explicitly , adding 
" __g " , " _go " , " ng_ " , and " g__ " ) . For sequences of words , 
the trigrams that can be generated from " the dog smelled like a skunk " 
are " # the dog " , " the dog smelled " , " dog smelled like ", " smelled 
like a " , " like a skunk " and " a skunk # " .""".split()
