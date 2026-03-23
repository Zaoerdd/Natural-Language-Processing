# Assignment 1 Report: Neural Network-based Text Classification


## 1. Task Overview
This assignment trains a neural network-based text classifier on the Chinese humor detection dataset. I implemented two tokenizers:

- **`basic_char`**: keeps only Chinese characters and treats each character as one token.
- **`advanced_jieba`**: segments Chinese spans into words with `jieba`, while preserving English words, numbers, and punctuations.

The classifier is based on `nn.EmbeddingBag` and a fully-connected network with two hidden layers ($128 \to 64$), which satisfies the assignment requirement.


## 2. Dataset Summary

| Split | Num Examples | Label 0 | Label 1 | Positive Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 12,677 | 9,031 | 3,646 | 0.2876 |
| **Test** | 651 | 481 | 170 | 0.2611 |


## 3. Vocabulary Comparison

| Tokenizer | Min Freq | Vocab Size | Observed Tokens | Avg Tokens per Example |
| :--- | :---: | :---: | :---: | :---: |
| `basic_char` | 2 | 2,182 | 202,194 | 17.72 |
| `advanced_jieba` | 2 | 5,576 | 150,779 | 13.22 |

*The advanced tokenizer produces a much larger vocabulary because it keeps richer token patterns such as Chinese multi-character words, English fragments, numbers, and punctuations.*

## 4. Test Results

| Tokenizer | Vocab Size | Accuracy | Precision | Recall | F1 | Macro F1 | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`advanced_jieba`** | **5,576** | **0.6728** | **0.3713** | **0.3647** | **0.3680** | **0.5736** | **0.8712** |
| `basic_char` | 2,182 | 0.6713 | 0.3608 | 0.3353 | 0.3476 | 0.5639 | 1.1252 |

**Best final model on the test set: `advanced_jieba`**


## 5. Discussion
In this experiment, the advanced tokenizer achieved slightly better overall test performance, while also substantially increasing vocabulary size. This suggests that richer tokenization can help the classifier capture additional useful patterns, but it comes with a larger vocabulary and therefore higher model complexity. However, the advanced tokenizer remains useful because it captures more linguistic structure and special patterns than pure character splitting.


## 6. Reproducibility
- **Random seed**: 42
- **Optimizer**: Adam
- **Learning rate**: 0.002
- **Batch size**: 64
- **Epochs**: 10
- **Validation split**: 10% of the provided training set
- **Device**: CPU