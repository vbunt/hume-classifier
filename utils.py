from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import random
from numpy import mean, std
from tqdm.notebook import tqdm as tqdm
import pickle

classes = ['dialogues', 'dissertation', 'enquiry', 'essay', 'history', 'political discourses', 'treatise']

stopwords = ['in', 'a', 'that', 'his', 'was',
             'by', 'which', 'he', 'it', 'with',
             'as', 'had', 'is', 'their', 'be',
             'this', 'for', 'from', 'but', 'were',
             'all', 'or', 'they', 'not', 'on',
             'him', 'any', 'an', 'them', 'so',
             'at', 'who', 'we', 'more', 'are',
             'these', 'have', 'no', 'her', "'s",
             'been', 'such', 'into', 'other', 'one',
             'than', 'would', 'some', 'every', 'though',
             'against', 'when', 'those', 'if', 'may',
             'our', 'most', 'same', 'i', 'even',
             'should', 'could', 'she', 'there', 'himself',
             'upon', 'only', 'can', 'without', 'must',
             'after', 'its', 'much', 'being', 'still',
             'first', 'us', 'what', 'where', 'will',
             'many', 'never', 'has', 'now', 'both',
             'might', 'during', 'themselves', 'before', 'ever',
             'under', 'among', 'therefore', 'over', 'yet',
             'cause', 'either', 'also', 'having', 'nor',
             'between', 'you', 'whom', 'each', 'another',
             'while', 'then', 'however', 'too', 'the',
             'to', 'of', 'and', 'thy', 'tis',
             'your', 'my', 'betwixt', 'cannot', 'ought']


id2label = {0: "dialogues", 
            1: "dissertation",
            2: "enquiry",
            3: "essay",
            4: "history",
            5: "political discourses",
            6: "treatise"}

label2id = {val : key for key, val in id2label.items()}

def import_dataset(my_seed=13, history_data='random'):
    df = pd.read_json('all.json')
    
    # ignore abstract, letter
    # take all dialogues, dissertation, essays, political discourses
    # choose 500 random paragraphs from history, treatise, enquiry
    
    if history_data == 'random':
        history_dataset = df[df.genre=='history'].sample(n=500, random_state=my_seed)
    elif history_data == 'automatic':
        history_dataset = pd.read_json('hume_json/history_automatic.json')
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


def get_scores(pred, y_test, title, matrix=True, print_=True, prec_rec=False):
    
    acc = accuracy_score(y_test, pred) * 100
    f1 = f1_score(y_test, pred, average='micro') * 100
    
    if print_:
        print(f"Accuracy: {acc:.2f}")
#         print(f"F1: {f1:.2f}")
        
    if prec_rec:
        s = precision_recall_fscore_support(y_test, pred)
        print(f'History precision: {s[0][4]*100:.2f}')
        print(f'History recall: {s[1][4]*100:.2f}')
#         print(f'History f-score: {s[2][4]:.2f}')
#         print(f'History support: {s[3][4]}')

    if matrix:
        fig, ax = plt.subplots(figsize=(10, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)

        cl = list(set(y_test) | set(pred))
        labels = [classes[i] for i in cl]

        ax.xaxis.set_ticks([i for i in range(0, len(cl))], labels, rotation=60)
        ax.yaxis.set_ticks([i for i in range(0, len(cl))], labels)
    
    return acc, f1
    
    
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
            

