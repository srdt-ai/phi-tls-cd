# generer des vecteurs de mots avec Word2Vec en Python 
  
# importe les modules
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
  
#  Lecture du fichier 'DOCTXT'
sample = open("C:\\Users\\Admin\\Desktop\\DOCTXT.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
  
data = [] 
  
# repete la procedure pour chaque phrase du fichier 
for i in sent_tokenize(f): 
    temp = [] 
      
    # isole les mots de la phrase 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
  
# creation du model CBOW
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
  
# impression des resultats 
print("Cosine similarity between 'EXEMPLE1' " + 
               "and 'EXEMPLE2' - CBOW : ", 
    model1.similarity('EXEMPLE1', 'EXEMPLE2')) 
      
print("Cosine similarity between 'EXEMPLE1' " +
                 "and 'EXEMPLE3' - CBOW : ", 
      model1.similarity('EXEMPLE1', 'EXEMPLE3')) 
  
# creation du modele Skip Gram 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 1) 
  
# impression des resultats
print("Cosine similarity between 'EXEMPLE1' " +
          "and 'EXEMPLE2' - Skip Gram : ", 
    model2.similarity('EXEMPLE1', 'EXEMPLE2')) 
      
print("Cosine similarity between 'EXEMPLE1' " +
            "and 'EXEMPLE3' - Skip Gram : ", 
      model2.similarity('EXEMPLE1, 'EXEMPLE3')) 