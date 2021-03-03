## Text Similarity Model : SUB-MODULES

The sub-modules of the text similarity model are as follows:

1. semantic_nn
2. semantic_bert
3. sequence_match based model
4. syntax_analysis using NLPSTANZA
5. topic_modelling approach- examining the topics and the distance between topics


#### Testing
`python3 text-sim-model/text_analysis.py`

```python
if __name__ == "__main__":
    sentence1,sentence2 = 'What can make python easy to learn?','How can you make python easy to learn?'
    print(TextAnalysis.text_analysis(sentence1,sentence2))
```
`RESULTS:`
```python
Semantic Analysis---Score: nn : 59.2659592628479, bert: 91.5 and label_flag assigned: is_similar
Sequence Match Analysis---Sequence Match Ratio :  84.93150684931507
Syntax Analysis:  {'What can make python easy to learn?': {'verb': 25.0, 'pron': 12.5, 'aux': 12.5, 'noun': 12.5, 'adj': 12.5, 'part': 12.5, 'punct': 12.5}, 'How can you make python easy to learn?': {'verb': 22.22, 'adv': 11.11, 'aux': 11.11, 'pron': 11.11, 'noun': 11.11, 'adj': 11.11, 'part': 11.11, 'punct': 11.11}}
Topic Modelling:  {'topics': ['learn', 'python', 'easi'], 'topic_distance': {'easi-learn': 3, 'easi-python': 6, 'learn-python': 5}}

# Function Return:-
('What can make python easy to learn?', 'How can you make python easy to learn?', 59.26,91.50, 'is_similar', 84.93150684931507, {'What can make python easy to learn?': {'verb': 25.0, 'pron': 12.5, 'aux': 12.5, 'noun': 12.5, 'adj': 12.5, 'part': 12.5, 'punct': 12.5}, 'How can you make python easy to learn?': {'verb': 22.22, 'adv': 11.11, 'aux': 11.11, 'pron': 11.11, 'noun': 11.11, 'adj': 11.11, 'part': 11.11, 'punct': 11.11}}, {'topics': ['learn', 'python', 'easi'], 'topic_distance': {'easi-learn': 3, 'easi-python': 6, 'learn-python': 5}})
```

### Author: Shivani Tyagi # Text-similarity
