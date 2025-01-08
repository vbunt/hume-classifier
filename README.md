# hume-classifier

David Hume is an 18th century Scottish philosopher and historian. It is believed that some of his political essays ended up becoming chapters in his "History of England". In my thesis I am trying to prove this hypothesis by training various ML models on genre-tagged data (genre classification task!). Eventually I will feed the entirety of Hume's "History" into a model to find all essay-like fragments. The results will hopefully shed some light on Hume's writing process. 


| model                   | fixed accuracy | mean accuracy | sd accuracy | time (on onetorch) |
| ----------------------- | -------------- | ------------- | ----------- | ------------------ |
| SVM                     | .738           | .739          | .023        | 5.749 |
| logreg                  | .751           | .745          | .019        | 1.859 |
| Gradient boosting       | .661           | -             | -           | 37.299 |
| Decision Tree           | .425           | -             | -           | 0.745 |
| Random forest           | .525           | -             | -           | 0.370 |
| distilbert-base-uncased | .761           | .762        | .027       | 1:04 (5 ep) |
| eccobert-base-cased-v1  | **.806**       | **.836**    | **.014**   | 4:04 (10 ep) |
| xlm-roberta-base        | .738           | .775        | .024       | 5:07 (10 ep) |

- *fixed accuracy* - accuracy on the fixed sample
- *mean + sd accuracy* - mean and standard deviation of accuracy on 15 randomised samples for train+test

