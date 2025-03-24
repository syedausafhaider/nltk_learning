import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet, genesis, senseval
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk, ngrams, FreqDist, ConditionalFreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.text import Text
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import wordnet as wn
from nltk.chunk import RegexpParser
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import CFG, ChartParser
import random
import matplotlib.pyplot as plt
import math

# Here I load a subset of the movie_reviews dataset (5 positive, 5 negative reviews)
print("=== Enhanced NLTK Features Demonstration with Movie Reviews Dataset ===")
pos_ids = movie_reviews.fileids('pos')[:5]
neg_ids = movie_reviews.fileids('neg')[:5]
text = " ".join(movie_reviews.raw(fileid) for fileid in pos_ids + neg_ids)
print(f"Dataset: Subset of NLTK Movie Reviews (5 pos + 5 neg reviews)")
print(f"Sample Text (first 200 characters):\n{text[:200]}...\n")

# Here I preprocess text into sentences and words
sentences = sent_tokenize(text)
words = word_tokenize(text)

# 1. Tokenization
print("1. Tokenization")
print("Number of Sentences:", len(sentences))
print("First Sentence:", sentences[0])
print("Number of Words:", len(words))
print("First 10 Words:", words[:10])
print()

# 2. Part-of-Speech (POS) Tagging
print("2. Part-of-Speech Tagging")
pos_tags = pos_tag(words, tagset='universal')
print("POS Tags (first 10):", pos_tags[:10])
print()

# 3. Stemming
print("3. Stemming")
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed Words (first 10):", stemmed_words[:10])
print()

# 4. Lemmatization
print("4. Lemmatization")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized Words (first 10):", lemmatized_words[:10])
print()

# 5. Stop Words Removal
print("5. Stop Words Removal")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Words after Stop Words Removal (first 10):", filtered_words[:10])
print()

# 6. Named Entity Recognition (NER)
print("6. Named Entity Recognition")
subset_pos_tags = pos_tag(word_tokenize(sentences[0]), tagset='universal')
ner_result = ne_chunk(subset_pos_tags)
print("NER Result (first sentence):\n", ner_result)
print()

# 7. Sentiment Analysis
print("7. Sentiment Analysis")
sid = SentimentIntensityAnalyzer()
sentiment_scores = sid.polarity_scores(text)
print("Sentiment Scores:", sentiment_scores)
print("Interpretation: Positive > 0.05 is positive, Negative < -0.05 is negative, else neutral")
print()

# 8. N-grams (Trigrams)
print("8. N-grams (Trigrams)")
trigrams = list(ngrams(words, 3))
print("Trigrams (first 5):", trigrams[:5])
print()

# 9. Frequency Distribution
print("9. Frequency Distribution")
freq_dist = FreqDist(words)
print("Most Common Words (top 5):", freq_dist.most_common(5))
print()

# 10. Concordance
print("10. Concordance for 'film'")
text_obj = Text(words)
print("Concordance:")
text_obj.concordance("film", lines=3)
print()

# 11. Collocations
print("11. Collocations")
text_obj.collocations(num=3)
print()

# 12. Synonyms and Antonyms
print("12. Synonyms and Antonyms (for 'great')")
synsets = wordnet.synsets("great")
synonyms = set()
antonyms = set()
for syn in synsets:
    for lemma in syn.lemmas():
        synonyms.add(lemma.name())
        if lemma.antonyms():
            antonyms.add(lemma.antonyms()[0].name())
print("Synonyms:", synonyms)
print("Antonyms:", antonyms if antonyms else "None found")
print()

# 13. Text Generation
print("13. Text Generation")
movie_text = Text(words)
print("Generated Text (10 words):")
movie_text.generate(words=10)
print()

# 14. Text Classification (Naive Bayes)
print("14. Text Classification (Naive Bayes)")
def word_features(words):
    return {word: True for word in words}
pos_reviews = [(word_features(movie_reviews.words(fileid)), 'pos') for fileid in pos_ids]
neg_reviews = [(word_features(movie_reviews.words(fileid)), 'neg') for fileid in neg_ids]
train_data = pos_reviews + neg_reviews
classifier = NaiveBayesClassifier.train(train_data)
test_words = word_tokenize(sentences[0])
print("Classification of first sentence:", classifier.classify(word_features(test_words)))
print("Most Informative Features:", classifier.show_most_informative_features(5))
print()

# 15. Chunking (Noun Phrases)
print("15. Chunking (Noun Phrases)")
grammar = "NP: {<DET>?<ADJ>*<NOUN>}"
cp = RegexpParser(grammar)
subset_pos_tags_full = pos_tag(word_tokenize(sentences[0]))
chunked = cp.parse(subset_pos_tags_full)
print("Chunked Result (first sentence):\n", chunked[:10], "...")
print()

# 16. Word Similarity (WordNet)
print("16. Word Similarity (between 'film' and 'movie')")
film_syn = wn.synset('film.n.01')
movie_syn = wn.synset('movie.n.01')
similarity = film_syn.wup_similarity(movie_syn)
print("Wu-Palmer Similarity:", similarity)
print()

# 17. Dispersion Plot
print("17. Dispersion Plot (for 'film', 'movie', 'great')")
plt.figure(figsize=(10, 2))
text_obj.dispersion_plot(["film", "movie", "great"])
plt.title("Dispersion Plot")
plt.show()
print("Note: Plot displayed in a separate window.")
print()

# 18. Conditional Frequency Distribution
print("18. Conditional Frequency Distribution (by POS tag)")
cfd = ConditionalFreqDist((tag, word) for (word, tag) in pos_tags)
print("Top 5 Nouns (NOUN):", cfd['NOUN'].most_common(5))
print("Top 5 Verbs (VERB):", cfd['VERB'].most_common(5))
print()

# 19. Text Entropy
print("19. Text Entropy")
freqs = [freq / len(words) for freq in freq_dist.values()]
entropy = -sum(p * math.log2(p) for p in freqs if p > 0)
print("Shannon Entropy of Text:", entropy)
print()

# 20. Lexical Diversity
print("20. Lexical Diversity")
lexical_diversity = len(set(words)) / len(words)
print("Lexical Diversity (unique words / total words):", lexical_diversity)
print()

# 21. Bigram Association Measures
print("21. Bigram Association Measures")
bigram_finder = BigramCollocationFinder.from_words(words)
bigram_finder.apply_freq_filter(3)  # Filter bigrams appearing < 3 times
top_bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, 5)
print("Top 5 Bigrams by Pointwise Mutual Information:", top_bigrams)
print()

# 22. Semantic Role Labeling (Simulated with POS and NER)
print("22. Semantic Role Labeling (Simulated)")
first_sent_words = word_tokenize(sentences[0])
first_sent_pos = pos_tag(first_sent_words)
roles = {"Agent": [], "Action": [], "Object": []}
for word, tag in first_sent_pos:
    if tag.startswith('NNP') or tag.startswith('NNS'):  # Nouns as potential agents/objects
        if "have" in roles["Action"]:  # After a verb like "have"
            roles["Object"].append(word)
        else:
            roles["Agent"].append(word)
    elif tag.startswith('VB'):  # Verbs as actions
        roles["Action"].append(word)
print("Simulated Roles for First Sentence:", roles)
print()

# 23. Word Sense Disambiguation (WSD)
print("23. Word Sense Disambiguation (for 'film')")
from nltk.wsd import lesk
sent = word_tokenize(sentences[0])
sense = lesk(sent, 'film')
print("Best Sense for 'film' in first sentence:", sense)
print("Definition:", sense.definition())
print()

# 24. Sentence Parsing with CFG
print("24. Sentence Parsing with CFG")
cfg = CFG.fromstring("""
    S -> NP VP
    NP -> DT NN | JJ NN | NN
    VP -> VB NP | VB
    DT -> 'the' | 'a'
    JJ -> 'comic' | 'plenty'
    NN -> 'films' | 'books' | 'success'
    VB -> 'adapted' | 'have'
""")
parser = ChartParser(cfg)
first_sent_short = word_tokenize("films adapted books")  # Simplified for demo
print("Parse Tree for 'films adapted books':")
for tree in parser.parse(first_sent_short):
    print(tree)
print()
