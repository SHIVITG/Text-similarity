## Text Similarity Using Siamese Deep Neural Network : SEMANTIC MODEL

Siamese neural network is a class of **neural network architectures that contain two or more** **identical** **subnetworks**. *identical* here means they have the same configuration with the same parameters 
and weights. Parameter updating is mirrored across both subnetworks.

It is a keras based implementation of deep siamese Bidirectional LSTM network to capture phrase/sentence similarity using word embeddings.

#### Install dependencies

`pip3 install -r requirements.txt`

## Text Similarity Model
 
 - In model/ : all the required scripts for training and testing that includes support files as well.
 - In data/ : the sample_data.csv that's the data used in training the model
 - In data/checkpoints : the saved pretrained model

### Usage

#### Training
`python3 model/train.py`

``` python
siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)

best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='data/')
```
#### Testing
`python3 text_sim.py`

```python
if __name__ == "__main__":
    sentence1,sentence2 = 'What can make python easy to learn?','How can you make python easy to learn?'
    textsim = TextSim(sentence1,sentence2)
    print(textsim.predict_score())
```
### References:

1. [Siamese Recurrent Architectures for Learning Sentence Similarity (2016)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)
2. Inspired from Tensorflow Implementation of  https://github.com/dhwajraj/deep-siamese-text-similarity

### Author: Shivani Tyagi
