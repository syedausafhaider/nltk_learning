# NLTK Features Demonstration with Movie Reviews Dataset

## Overview
This Python script demonstrates a wide range of features from the Natural Language Toolkit (NLTK) using the `movie_reviews` corpus. Designed as an educational tool, it showcases both basic and advanced NLP functionalities, from tokenization to syntactic parsing. The script processes a subset of movie reviews (5 positive and 5 negative) and applies various NLP analyses.

## Purpose
The primary goal of this script is to provide a hands-on example of NLTK's capabilities, covering:
- Text preprocessing
- Statistical analysis
- Semantic understanding
- Syntactic parsing

Each feature is clearly labeled, and comments with "Here I ..." statements guide users through the process.

## Prerequisites
Before running the script, ensure the required dependencies and NLTK datasets are installed:

### Install Dependencies
```bash
pip install nltk matplotlib
```

### Download NLTK Data
```python
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.download('book')
nltk.download('senseval')
nltk.download('universal_tagset')
```

## Features Demonstrated
The script includes 24 distinct NLTK features applied to the movie reviews dataset:

1. **Tokenization** - Sentence and word tokenization with `sent_tokenize` and `word_tokenize`.
2. **Part-of-Speech (POS) Tagging** - Assigns grammatical tags using `pos_tag` (universal tagset).
3. **Stemming** - Reduces words to their root form using `PorterStemmer`.
4. **Lemmatization** - Reduces words to their base form with `WordNetLemmatizer`.
5. **Stop Words Removal** - Filters out common words using `stopwords`.
6. **Named Entity Recognition (NER)** - Identifies entities (people, places) using `ne_chunk`.
7. **Sentiment Analysis** - Analyzes text sentiment with `VADER SentimentIntensityAnalyzer`.
8. **N-grams (Trigrams)** - Generates 3-word sequences with `ngrams`.
9. **Frequency Distribution** - Counts word occurrences using `FreqDist`.
10. **Concordance** - Shows word contexts using `Text.concordance`.
11. **Collocations** - Finds frequent word pairs with `Text.collocations`.
12. **Synonyms and Antonyms** - Retrieves word variations using `WordNet synsets`.
13. **Text Generation** - Creates random text based on patterns with `Text.generate`.
14. **Text Classification (Naive Bayes)** - Classifies text as positive or negative using `NaiveBayesClassifier`.
15. **Chunking (Noun Phrases)** - Identifies noun phrases with `RegexpParser`.
16. **Word Similarity (WordNet)** - Measures semantic similarity with `wup_similarity`.
17. **Dispersion Plot** - Visualizes word distribution with `Text.dispersion_plot` (requires `matplotlib`).
18. **Conditional Frequency Distribution** - Analyzes word frequencies by POS tag using `ConditionalFreqDist`.
19. **Text Entropy** - Calculates Shannon entropy to measure text unpredictability.
20. **Lexical Diversity** - Computes the ratio of unique words to total words.
21. **Bigram Association Measures** - Finds significant bigrams using `BigramCollocationFinder` and PMI.
22. **Semantic Role Labeling (Simulated)** - Approximates agent-action-object roles using POS tags.
23. **Word Sense Disambiguation (WSD)** - Disambiguates word meanings with the Lesk algorithm.
24. **Sentence Parsing with CFG** - Parses sentence structure using a context-free grammar and `ChartParser`.

## How It Works
1. **Dataset:** Loads 5 positive and 5 negative reviews from `movie_reviews` and concatenates them into a single text string.
2. **Preprocessing:** Tokenizes the text into sentences and words for analysis.
3. **Feature Execution:** Applies each of the 24 features and prints results with descriptive labels.
4. **Visualization:** Generates a dispersion plot in a separate window using `matplotlib`.

## Usage
1. Save the script as `nltk_demo.py`.
2. Ensure all prerequisites are met.
3. Run the script:

```bash
python nltk_learning.py
```

4. Review the console output for text analysis results and check the dispersion plot in a new window.

## Sample Output
```
=== Enhanced NLTK Features Demonstration with Movie Reviews Dataset ===
Dataset: Subset of NLTK Movie Reviews (5 pos + 5 neg reviews)

Sample Text (first 200 characters):
films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before . ...

1. Tokenization
Number of Sentences: 147
First Sentence: films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before .
Number of Words: 3229
First 10 Words: ['films', 'adapted', 'from', 'comic', 'books', 'have', 'had', 'plenty', 'of', 'success']

...

24. Sentence Parsing with CFG
Parse Tree for 'films adapted books':
(S (NP (NN films)) (VP (VB adapted) (NP (NN books))))
```

## Notes
- **Performance:** Features like NER, classification, and parsing are applied to small subsets for efficiency.
- **Customization:** Modify `pos_ids` and `neg_ids` to use more reviews or switch to another NLTK corpus (e.g., `brown`).
- **Limitations:** Some features (e.g., SRL, CFG parsing) are simplified. For advanced implementations, consider external tools like `spaCy` or `Stanford NLP`.
- **Visualization:** `matplotlib` is required for the dispersion plot. Ensure it’s installed and a graphical environment is available.

## Learning Opportunities
- Experiment with different NLTK corpora or custom text files.
- Expand the CFG for more complex sentence parsing.
- Enhance the classifier with more training data or additional features.
- Explore additional NLTK modules (e.g., `nltk.parse`, `nltk.translate`).

This script serves as a starting point for mastering NLTK, and users are encouraged to extend it for specific NLP projects or research.

---

Happy coding! 🚀

