
# coding: utf-8

import numpy as np
from w2v_utils import *

# chargement du doc (words = ensemble de mots du vocabulaire)) en définissant via GloVe le nombre N de dimensions souhaitees

words, word_to_vec_map = read_glove_vecs('../../EXEMPLE_FICHIER_DOC.Nd.txt')

def cosine_similarity(u, v):
    """
    Cosine similarity = exprime le degré de similarité entre u et v
    """
    
    distance = 0.0
    
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot / np.dot(norm_u, norm_v)
    
    return cosine_similarity

EXEMPLE1 = word_to_vec_map["EXEMPLE1"]
EXEMPLE2 = word_to_vec_map["EXEMPLE2"]
EXEMPLE3 = word_to_vec_map["EXEMPLE3"]
EXEMPLE4 = word_to_vec_map["EXEMPLE4"]
EXEMPLE5 = word_to_vec_map["EXEMPLE5"]
EXEMPLE6 = word_to_vec_map["EXEMPLE6"]
EXEMPLE7 = word_to_vec_map["EXEMPLE7"]
EXEMPLE8 = word_to_vec_map["EXEMPLE8"]

print("cosine_similarity(EXEMPLE1, EXEMPLE2) = ", cosine_similarity(EXEMPLE1, EXEMPLE2))
print("cosine_similarity(EXEMPLE3, EXEMPLE4) = ",cosine_similarity(EXEMPLE3, EXEMPLE4))
print("cosine_similarity(EXEMPLE5 - EXEMPLE6, EXEMPLE7 - EXEMPLE8) = ",cosine_similarity(EXEMPLE5 - EXEMPLE6, EXEMPLE7 - EXEMPLE8))


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    présentation des résultats sous forme d'analogie sémantique
    """
    # conversion en minuscules
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              
    best_word = None                   

    # début de la boucle sur l'ensemble des mots vectorisés
    for w in words:        
    
        if w in [word_a, word_b, word_c] :
            continue
        
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[w] - e_c))
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word

 """
algorithme de débiaisage des co-occurrences et rapprochements dus au genre des mots (ex. Père/Mère = genre, etc.)
    """

def neutralize(word, g, word_to_vec_map):
   
    e = word_to_vec_map[word]
    e_biascomponent = np.dot(e ,g) / np.sum(g * g) * g
    e_debiased = e - e_biascomponent
    
    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
   
  
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1],word_to_vec_map[w2]
    
    mu = (e_w1 + e_w2) / 2

    mu_B = np.dot(mu, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    mu_orth = mu - mu_B
    
    e_w1B = np.dot(e_w1, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
        
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth * mu_orth))) * (e_w1B - mu_B) / np.linalg.norm(e_w1 - mu_orth - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth * mu_orth))) * (e_w2B - mu_B) / np.linalg.norm(e_w2 - mu_orth - mu_B)


    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
                                                                
    
    return e1, e2


