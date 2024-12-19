# hume-classifier

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


