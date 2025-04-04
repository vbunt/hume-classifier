from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import random
from numpy import mean, std
from tqdm.notebook import tqdm as tqdm
import pickle

classes = ['dialogues', 'dissertation', 'enquiry', 'essay', 'history', 'political discourses', 'treatise']

id2label = {0: "dialogues", 
            1: "dissertation",
            2: "enquiry",
            3: "essay",
            4: "history",
            5: "political discourses",
            6: "treatise"}

label2id = {val : key for key, val in id2label.items()}

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
        print(f"Accuracy: {accuracy_score(y_test,pred)}%")

    if matrix:
        fig, ax = plt.subplots(figsize=(10, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        
        left = min(min(y_test), min(pred))
        right = max(max(y_test), max(pred))
        ax.xaxis.set_ticks([i for i in range(0, right+1-left)], classes[left:right+1], rotation=60)
        ax.yaxis.set_ticks([i for i in range(0, right+1-left)], classes[left:right+1])
        plt.title(title)
    
    return accuracy_score(y_test,pred)

def run_once(model):
    seed = random.randint(1, 100)
    dataset = import_dataset(seed)
    
    y_train = dataset['train']['labels']
    y_test = dataset['test']['labels']
    
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(dataset['train']['text'])
    x_test = vectorizer.transform(dataset['test']['text'])
    
    
    model.fit(x_train,y_train)
    svm_pred = model.predict(x_test)

    accuracy = get_scores(svm_pred, y_test, '_', matrix=False)
    
    return accuracy

def run_many(model, n):
    accuracy_scores = []
    for i in tqdm(range(n)):
        accuracy_scores.append(run_once(model))
        
    print('Mean accuracy: ', mean(accuracy_scores))
    print('Standard deviation for accuracy: ', std(accuracy_scores))
    
    
def feature_extraction(model_path, vect_path, name, n=20):
    
    if isinstance(model_path, str):
        with open(model_path, 'rb') as f:
            model = pickle.load(f) 
    else:
        model = model_path
        
    if isinstance(vect_path, str):
        with open(vect_path, 'rb') as f:
            vect = pickle.load(f)           
    else:
        vect = vect_path
            
    
    feature_names = list(vect.get_feature_names_out())
    
    with open(f'model_features/{name}.txt', 'w') as file:
        
        for genre in label2id.keys():
        
            ind = label2id[genre]

            zips = [(name, coef) for name, coef in zip(feature_names, model.coef_[ind])]
            
            file.write(f'{genre} | positive')
            file.write('\n')

            for word in sorted(zips, key = lambda pair: pair[1], reverse=True)[:n]:
                file.write(f'{word[0]}    {word[1]:.2f}')
                file.write('\n')
                
            file.write('\n')
            file.write('\n')

            file.write(f'{genre} | negative')
            file.write('\n')

            for word in sorted(zips, key = lambda pair: pair[1])[:n]:
                file.write(f'{word[0]}    {word[1]:.2f}')
                file.write('\n')

            file.write('\n')
            file.write('\n')
            

