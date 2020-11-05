import spacy 
from spacy.lang.te import Telugu
nlp = Telugu()  # use directly
nlp = spacy.blank("te")
# Pkgs for Normalizing Text
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# Import Heapq for Finding the Top N Sentences
from heapq import nlargest
from indicnlp.tokenize import sentence_tokenize


def Text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency # word.text is tokenization in spacy
    word_frequencies = {}  
    for word in docx:  
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentences=sentence_tokenize.sentence_split(docx, lang='tel')
    from sklearn.feature_extraction.text import CountVectorizer
    #create vectorizer object
    c = CountVectorizer()
    bow_matrix = c.fit_transform(sentences)
    vector=c.transform(sentences)
    from sklearn.feature_extraction.text import TfidfTransformer
    normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
    similarity_graph = normalized_matrix * normalized_matrix.T
    import networkx as nx
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    sentence_array = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    freq_max =float(sentence_array[0][0])
    freq_min=float(sentence_array[len(sentence_array) - 1][0])
    temp_array = []
    for i in range(0,len(sentence_array)):
        if freq_max - freq_min == 0:
            temp_array.append(0)
        else:
            temp_array.append((float(sentence_array[i][0]) - freq_min)/(freq_max - freq_min))
            threshold = (sum(temp_array) /len(temp_array)) + 0.25
            seq_list = []
            for i in range(0,len(temp_array)):
                if temp_array[i] > threshold:
                        sentence_list.append(sentence_array[i][1])
                        for sentence in sentences:
                            if sentence in sentence_list:
                                seq_list.append(sentence)
                                summary = seq_list
                                return summary