from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm


classes = ['dialogues', 'dissertation', 'enquiry', 'essay', 'history', 'political discourses', 'treatise']

def import_dataset(my_seed=13):
    df = pd.read_json('all.json')
    
    # ignore abstract, letter
    # take all dialogues, dissertation, essays, political discourses
    # choose 500 random paragraphs from history, treatise, enquiry

    history_dataset = df[df.genre=='history'].sample(n=500, random_state=my_seed)
    treatise_dataset = df[df.genre=='treatise'].sample(n=500, random_state=my_seed)
    enquiry_dataset = df[df.genre=='enquiry'].sample(n=500, random_state=my_seed)

    dialogues_dataset = df[df.genre=='dialogues']
    dissertation_dataset = df[df.genre=='dissertation']
    essay_dataset = df[df.genre=='essay']
    discourses_dataset = df[df.genre=='political discourses']

    # concatenate all datasets into one

    combined_dataset = pd.concat([history_dataset, 
                        treatise_dataset, 
                        dialogues_dataset, 
                        dissertation_dataset, 
                        essay_dataset,
                        enquiry_dataset,
                        discourses_dataset], ignore_index=True)
    
    # split into three

    unsplit_dataset = Dataset.from_pandas(combined_dataset)

    train_test_dataset = unsplit_dataset.train_test_split(test_size=0.2, shuffle=True, seed=my_seed)

    test_eval_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=my_seed)

    dataset = DatasetDict({'train' : train_test_dataset['train'], 
                           'eval' : test_eval_dataset['train'],
                           'test' : test_eval_dataset['test']})
    
    return dataset


def get_scores(pred, y_test, title, matrix=True, print_=True):
    if print_:
        print(f"Accuracy: {accuracy_score(y_test,pred) * 100}%")

    if matrix:
        fig, ax = plt.subplots(figsize=(10, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        ax.xaxis.set_ticks([0, 1, 2, 3, 4, 5, 6], classes, rotation=60)
        ax.yaxis.set_ticks([0, 1, 2, 3, 4, 5, 6], classes)
        plt.title(title)
    
    return accuracy_score(y_test,pred) * 100

