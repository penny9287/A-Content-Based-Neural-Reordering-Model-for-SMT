# A-content-based-neural-reordering-model

Simplified implementation of "A Content-Based Neural Reordering Model for Statistical Machine Translation" paper
# Abstract
Phrase-based lexicalized reordering models have attracted extensive interest in statistical machine translation (SMT) due to their capacity for dealing with swap between consecutive phrases. However, translations between two languages that with significant differences in syntactic structure have made it challenging to generate a semantically and syntactically correct word sequence. In an effort to alleviate this problem, we propose a novel content-based neural reordering model that estimates reordering probabilities based on the words of its surrounding contexts. We first utilize a simple convolutional neural network (CNN) to capture semantic contents conditioned on various sizes of context. And then we employ a softmax layer to predict the reordering orientations and probability distributions. Experimental results show that our model provides statistically obvious improvements for both Chinese-Uyghur (+0.48 on CWMT2015) and Chinese-English (+0.27 on CWMT2013) translation tasks over conventional lexicalized reordering models.
# Link
https://doi.org/10.1007/978-981-10-7134-8_11
# Usage
* Install Keras
* Run "python train.py --data_dir=path_to_data --embedding_file_path=path_to_embedding --model_name=name_of_model"
