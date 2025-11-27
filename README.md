# DL-Forward Approaches to David Hume’s _History of England_
This is a short and informal description of my Master's thesis. The thesis was written at the University of Helsinki in 2024-2025 under careful supervision of Mikko Tolonen and Maciej Janicki. 
Full text: http://hdl.handle.net/10138/598293

## General Premise

David Hume has spent most of his years writing philosophical works, such as essays and treatises. Yet, at the end of his career he published a six-volume *History of England*. The shift from one genre to another was not clean: historians have long known Hume's *History* to contain non-narrative fragments of philosophical writing. The goal of this work is to create a model that could automatically extract such fragments. First I train a multiclass classifier model that assigns a genre label to a paragraph. Then, I process the entire text of _ History_ and use the model's logits as probability scores of a given paragraph belonging to each of the genres. Then I extract sequences of paragraphs where the scores for a genre are increased, compared to the rest of the text. Then I evaluate these sequences qualitatively.

## Do We Know What _Genre_ Is?

Literary scholars do not know what exactly constitutes a genre, but for the purposes of this work, *genre* is the title that Hume chose for a particular work. For example, if a text was published in the collection titled "Essays, Moral and Political", then its genre, and the label for the model, is 'essay'.

Of course, the entire text of _The History of England_ would then have to be tagged as 'history'. However, since I assume that some historical paragraphs are, actually, written in other genres and should be tagged as such, it is important to train the classifier on the paragraphs that are as representative of classical history as possible. To choose such paragraphs, I trained a Logistic regression model and chose the samples where the logit for 'history' class was >0.95. (In the thesis, I did this after the first batch of experiments, when I realized that the models were mislabelling 'history' class.)

## The Data

*Genre dataset* includes samples from all genres present in Hume's writing. Each sample is a paragraph. For longer works, I used only 500 samples to keep the data relatively balanced. 

| Genre        | N. of samples in data | N. of samples in _Genre dataset_  |
| ------------ | --------------------- | ------------------------------- |
| dialogues    | 239                   | 239                             |
| dissertation | 318                   | 318                             |
| enquiry      | 632                   | 500                             |
| essay        | 555                   | 555                             |
| political discourse | 483            | 483                             |
| treatise     | 121                   | 500                             |
| history      | 5045                  | 310 (chosen via classifier, see above) |
| Total        | 8490                  |  2905                           |

*Essay-in-history dataset (EH dataset)* consists of 66 samples from _The History of England_ that have essay-like qualities: they are nonnarrative and consist of the author’s thoughts on a general subject. They were hand-picked by my supervisor. The model should classify these samples as any philosophical genre, i.e. any genre but 'history'. 

## Setup
#### Models
- Logistic regression
- SVM
- distilbert-base-uncased
- xlm-roberta-base
- eccobert-base-cased-v1, which is a bert-base-uncased model trained on 18th century data. 

I expected that Transformer models would outperform everything else. Training a non-Transformer NN didn't seem viable effort-vise. I chose to experiment with classic ML models on the off-chance that a simple model would provide decent results with very little computational effort. 
#### Features
I used tf-idf features for Logistic regression and SVM, and native Transformer features for Transformers. 
#### Metrics
In multi-class problems, F-score is simply accuracy (correct / samples). I additionally used confusion matrices to analyse mistakes. 
#### Training 
Transformers were trained for 10 epochs, learning rate = 0.00001, weight decay = 0.01, with early stopping employed after 2 epochs with no positive change. Train/test/eval split was 80/10/10. Classic ML models used train/test.

## Experiments

First, I train/fine-tune the models on Genre dataset. The models learn to differentiate between 7 genres. Task 1 tests their ability to classify 7 genres. Task 2 tests their ability to classify EH dataset as anything but 'history'. 

| Model                       | Task 1: Classifying 7 genres, accuracy | Task 2: Classifying EH dataset, accuracy |
| --------------------------- | -------------------------------------- | ---------------------------------------- |
| Logistic regression                | 76.29                                  | 74.63                                    |
| SVM                         | 81.79                             | 71.64                                    |
| distilbert-base-uncased | 79.73                                  | **88.06**                                |
| eccobert-base-cased-v1  | **87.63**                              | 82.09                                |
| xlm-roberta-base            | 74.57                                  | 64.18                                    |

Interestingly, classic ML models show results on par with Transformers, while XLM-RoBERTa surprisingly comes last in both tasks.  

I chose distilbert-base-uncased for futher experiments, because it showed the best result on classifying EH dataset. In hindsight, it is unclear which improvements are statistically significant. EH dataset includes only 66 samples, so the difference in accuracy between DistilBERT and ECCO-BERT is 4 samples. Perhaps, I should have prioritized the performance on task 1, and chosen ECCO-BERT instead. 

#### Mechanism of Burst Extraction

The model is applied to _The History of England_, paragraph by paragraph. For each of them the model returns seven numbers: the probabilities of a paragraph belonging to each of the genres.

The method is inspired by the one used by Zhang et al. (2022), although with some modifications. The authors use only the most probable genre of a paragraph, instead of probabilities of all genres. This approach does not work in the Hume case, because the classifier predicts ‘history’ for the overwhelming majority of paragraphs. However, the probabilities of ‘history’ are not constant: sometimes they drop. Of interest are paragraphs where the probability of other genres is higher than in other parts of the book. If the probability is consistently increased, the part might belong to a different genre.

To find such parts I am using the Kleinberg algorithm (Kleinberg, 2002) for burst detection. The algorithm considers the number of _target events_ among _all events_, and if the number of
target events suddenly increases, it counts the length of the increase as a _burst_. Unfortunately, the algorithm is not designed to work with increasing probabilities, so I had to adapt my data.

The algorithm runs separately for each genre. Every paragraph is an _event_. The lists of target events and all events are equal to each other in length and to the number of considered
paragraphs, which is the length of the book. The list of all events is a list of 1s. The list of target events is a list of 1s and 0s. If the probability of the paragraph belonging to a specific genre is higher than threshold (I chose threshold=0.15), this paragraph is included as a 1, otherwise as a 0.

Unfortunately, this approach overlooks how much higher the value is, compared to the threshold. The probabilities get flattened, which leads to less precise extractions. There is
room for improvement in my application of the method.

## Assessment

After burst extraction, a volume looks like this: <img width="1511" height="511" alt="6 volume_2" src="https://github.com/user-attachments/assets/5abe6fa8-4c50-49cf-998a-d766f979a6f6" />
The x-axis shows paragraphs and chapter boundaries; the y-axis shows probabilities. Each dot in the scatterplot represents a probability of a paragraph belonging to a specific genre. The bars are bursts found by the Kleinberg algorithm. The colors of dots and bars represent genres.

In general, the findings correspond to the philologists' expectations about the text.
1. The volumes that were written first are more philosophical than the later ones, as shown here: <img width="393" height="308" alt="6_burst_percentage" src="https://github.com/user-attachments/assets/4ebb48db-747a-4a7f-9562-32577b6be901" />
2. The most common non-historical genres are 'essay' and 'political discourse' (meaning political and economical essays)
3. Long extracted bursts make sense. For example, in Volume 2 (above), every chapter ends on a 'political discourse' burst, which correspond to sections where Hume discusses the economy of a period. 
4. Short extracted bursts are often paragraphs that are neither clearly historical or clearly philosophical, but a secret third thing, such as non-narrative descriptions.

The assessment shows that only some of the extracted bursts are of philosophical genres. More often than not, they show parts of *History* that diverge from the classical style, but do not necessarily step into the realm of philosophical writing. The model is thus a compass, not a map; it only points into a possible direction, and it is the researcher’s job to investigate it.

## What I Would Do Differently If I Ever Do a PHD About This
1. Gain a better understanding of my data before doing the experiments: read more secondary literature, find specific observations to test, try to find essay-like parts myself.
2. Apply statistical testing to determine if models perform significantly better or not. 
3. Apply a better burst extraction algorithm. As I have written above, the current approach loses vital information, which leads to noise in extracted bursts. 

## What Else I Would Do If I Ever Do a PHD About This
1. Apply the pipeline to authors beyond Hume. I was told there are questions to be answered in the realm of Adam Smith, although I did not look into him myself. 
2. Try clustering. Maybe some philosophical genres are very close to each other, and it does not really make sense to separate them for our purposes? Maybe philosophical fragments in _History_ would cluster with other philosophical works right away? 
3. Experiment with models that have a bigger context window and try to classify sections instead of paragraphs.

## References
1. Kleinberg, J. (2002). Bursty and hierarchical structure in streams. *Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining*, 91– 101.
2. Zhang, J., Ryan, Y. C., Rastas, I., Ginter, F., Tolonen, M., & Babbar, R. (2022). Detecting sequential genre change in eighteenth-century texts. *CEUR Workshop Proceedings*, 3290, 243–255.
