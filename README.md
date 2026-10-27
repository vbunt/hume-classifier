# DL-Forward Approaches to David Hume’s _History of England_
This is a short and informal description of my Master's thesis. The thesis was written at the University of Helsinki in 2024-2025 under careful supervision of Mikko Tolonen and Maciej Janicki. 
October 2026: Added independently done input attribution experiment. 

Full text: http://hdl.handle.net/10138/598293

## General Premise

David Hume has spent most of his years writing philosophical works, such as essays and treatises. Yet, at the end of his career he published a six-volume *History of England*. The shift from one genre to another was not clean: historians have long known Hume's *History* to contain non-narrative fragments of philosophical writing. The goal of this work is to create a model that could automatically extract such fragments. First I train a multiclass classifier model that assigns a genre label to a paragraph. Then, I process the entire text of _History_ and use the model's logits as probability scores of a given paragraph belonging to each of the genres. Then I extract sequences of paragraphs where the scores for a genre are increased, compared to the rest of the text. Then I evaluate these sequences qualitatively.

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

## Input Attribution (done in 2026)

I was interested in how the classifier reached its predictions. This question can be considered in terms of model insides or in terms of input strings. In the first case, we can look at what neurons light up for each sample or how the prediction changes as it travels through the model. This would be attributing the result to different model components via approaches such as Direct Logit Attribution. In the second case, we can see which input tokens drive the prediction towards different classes. This is called input attribution, and it is the approach I am using here. 

First, I want to learn which tokens on average drive the prediction towards specific classes. Second, I want to look at attributions for specific paragraphs and compare it to my judgement on them from the thesis.

I use [Integrated gradients](https://captum.ai/docs/extension/integrated_gradients). The method works by computing a gradient on the path between a default token, such as UNK, and the token for which attribution is done, with regards to the probability of a specific class. 

I apply input attribution to 297 paragraphs from the test set of my Hume dataset. For each sample, attribution is computed for each possible class for each token. This ensures that I have the same number of computations for each class and can compare token attribution scores between classes. I average scores for every token for every class, which allows to compare tokens attribution scores across one class. 

Words that are split between multiple tokens were joined into one. Resulting word's attribution score equals to the highest one among the scores for the word's parts. 

I report results only for 'essay', 'history' and 'political discourses' labels as classes that are predicted most often. The table shows top-10 tokens for each genre that are used at least 5 times in the test set:

| Essay      |          | History      |          | Political discourses |          |
| ---------- | -------- | ------------ | -------- | -------------------- | -------- |
| **token**  | **attr** | **token**    | **attr** | **token**            | **attr** |
| politics   | 0.31729  | english      | 0.28456  | slaves               | 0.28019  |
| ancients   | 0.26799  | england      | 0.17325  | demea                | 0.26203  |
| arts       | 0.26697  | king         | 0.17193  | land                 | 0.25336  |
| poly       | 0.2634   | parliament   | 0.16616  | cleanthes            | 0.25216  |
| nation     | 0.26214  | henry        | 0.15015  | taxes                | 0.24277  |
| pagan      | 0.24249  | charles      | 0.14685  | ##the                | 0.23314  |
| manners    | 0.23012  | commonwealth | 0.14499  | money                | 0.233    |
| polite     | 0.22005  | york         | 0.13306  | roman                | 0.23061  |
| government | 0.21391  | lord         | 0.12648  | senate               | 0.22282  |
| fury       | 0.19474  | scotland     | 0.12636  | clean                | 0.22222  |

There are only so many things that separate tokens can tell about a class; in this case, we lose syntactic and morphological information that would probably just as useful at distinguishing history from philosophical genres. The current analysis shows that classes differ on topics: 'history' passages talk about specific places and people, 'political discourses' cover politics and economy, 'essay' includes a variety of philosophical topics, such as politics, arts, religion (token "poly" is the fist part of *polytheism* and *polytheist*), which corresponds with the broad range of topics in Hume's essays. In the thesis, I looked at weights for a Logistic Regression classifier, and observed a similar token and topic distribution. 

### Specific Examples

In the thesis, I analyse specific bursts to see if they are, indeed, philosophical. The first example I show is a 1-paragraph burst with critique of Oliver Cromwell's government:

> A government, totally military and despotic, is almost sure, after some time, to fall into impotence and languor: But when it immediately succeeds a legal constitution, it may, at first, to foreign nations appear very vigorous and active, and may exert with more unanimity that power, spirit, and riches, which had been acquired under a better form. (H 61.29)

The model predicts class 3 ('essay'), supported by "government", "military", "foreign nations", "legal constitution", "power" and "spirit". 
<img width="880" height="171" alt="Screenshot from 2026-10-27 11-33-44" src="https://github.com/user-attachments/assets/d2064a3d-6d91-4fb0-b2b4-006c5540b278" />

At the same time, "government", "military" and "power" weigh the prediction towards class 4 ('history), but many more tokens move it away from it, most notably, "constitution" and "is". It did not come up in the IG analisys, but in the thesis similar analysis of function words showed that 'history' tends to use past tense, while philosophical genres use present tense. It is nice to see the observation made on Logistic Regression still be true for BERT. 
<img width="880" height="171" alt="Screenshot from 2026-10-27 11-33-53" src="https://github.com/user-attachments/assets/a5416d0b-6a78-47fd-9ac9-d45c21aa273d" />


Hume starts the first volume of History with a long description of the Britons and various aspects of their culture. This description got extracted almost in its entirety. This is one of its paragraphs:

> The Britons were divided into many small nations or tribes; and being a military people, whose sole property was their arms and their cattle, it was impossible, after they had acquired a relish of liberty, for their princes or chieftains to establish any despotic authority over them. Their governments, though monarchical, were free, as well as those of all the Celtic nations; and the common people seem even to have enjoyed more liberty among them, than among the nations of Gaul, from whom they were descended. Each state was divided into factions within itself. It was agitated with jealousy or animosity against the neighbouring states: And while the arts of peace were yet unknown, wars were the chief occupation, and formed the chief object of ambition, among the people. (H 1.3)

I would not say that it is philosophical, but it is a description, not a narrative. It was classified as class 5 ('political discourses')

It is pulled towards 'history' by "princes", "chieftains" and "monarch" (which continue the trend of political titles seen in top-10 for the genre), and "britons".
<img width="880" height="246" alt="Screenshot from 2026-10-27 11-29-44" src="https://github.com/user-attachments/assets/305c5f3f-e66a-4bed-b0fb-566e58202460" />

Interestingly enough, the reason for labeling it as 'political discourses' seems to be the other tribes: "gaul", "celtic" and "people" have high attribution scores for the genre. This curious genre division was already caught on in the thesis where I speculated that "it probably got classified as ‘political discourse’ due to Hume’s many discussions of ancient examples in his political essays". 


The last two passages come from an 'essay' burst of 12 paragraphs in chapter 62. Hume introduces this part as a “general survey of the age, so far as regards manners, finances, arms, commerce, arts and sciences”. As I write in the thesis, these are "Hume’s critical descriptions of the main writers of the age. While it is not essay-like per se, it is undoubtedly the author’s personal opinions". The model, however, picks up not opinion words, like "the greatest genius", "admirable", "disagreeable", "extremely corrupted by the bad taste", "no ear for harmony", but on mentions of arts: "poems", "poet", "poetry", "verses", "writings", "prose", "rhyme". IG do also manage to pick up "extremely corrupted by the bad taste". 

> It is however remarkable, that the greatest genius by far, that shone out in England during this period, was deeply engaged with these fanatics, and even prostituted his pen in theological controversy, in factious disputes, and in justifying the most violent measures of the party. This was John Milton, whose poems are admirable, though liable to some objections; his prose writings disagreeable, though not altogether defective in genius. (H 62.93)

Attribution for 'history':
<img width="880" height="269" alt="Screenshot from 2026-10-27 11-27-28" src="https://github.com/user-attachments/assets/8641eff4-e031-4e94-bc75-150a252074ec" />

Attribution for 'essay':
<img width="880" height="269" alt="Screenshot from 2026-10-27 11-27-19" src="https://github.com/user-attachments/assets/a13f6c0a-a6cf-4745-b33c-a6153284e1ad" />

> Cowley is an author extremely corrupted by the bad taste of his age; but had he lived even in the purest times of Greece or Rome, he must always have been a very indifferent poet. He had no ear for harmony; and his verses are only known to be such by the rhyme, which terminates them. (H 62.98)

Attribution for 'history':
<img width="880" height="400" alt="Screenshot from 2026-10-27 11-25-59" src="https://github.com/user-attachments/assets/95aec910-4f45-4b0c-9bb0-fb4fce406524" />

Attribution for 'essay':
<img width="880" height="400" alt="Screenshot from 2026-10-27 11-25-48" src="https://github.com/user-attachments/assets/48b50ea6-0afc-4511-be81-ff527c5cb84e" />

The next experiment could be replacing all mentions of poetry with mentions of military to see if the model's prediction would change. 

In general, the study of token attributions supports different claims I make in the thesis and repeats some findings from the Linear Regression study.  


## What I Would Do Differently If I Ever Do a PhD About This
1. Gain a better understanding of my data before doing the experiments: read more secondary literature, find specific observations to test, try to find essay-like parts myself.
2. Apply statistical testing to determine if models perform significantly better or not. 
3. Apply a better burst extraction algorithm. As I have written above, the current approach loses vital information, which leads to noise in extracted bursts. 

## What Else I Would Do If I Ever Do a PhD About This
1. Apply the pipeline to authors beyond Hume. I was told there are questions to be answered in the realm of Adam Smith, although I did not look into him myself. 
2. Try clustering. Maybe some philosophical genres are very close to each other, and it does not really make sense to separate them for our purposes? Maybe philosophical fragments in _History_ would cluster with other philosophical works right away? 
3. Experiment with models that have a bigger context window and try to classify sections instead of paragraphs.

## References
1. Kleinberg, J. (2002). Bursty and hierarchical structure in streams. *Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining*, 91– 101.
2. Zhang, J., Ryan, Y. C., Rastas, I., Ginter, F., Tolonen, M., & Babbar, R. (2022). Detecting sequential genre change in eighteenth-century texts. *CEUR Workshop Proceedings*, 3290, 243–255.
