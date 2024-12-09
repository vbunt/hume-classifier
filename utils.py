import pandas as pd
from datasets import Dataset, DatasetDict

def import_dataset():
    df = pd.read_json('all.json')
    
    # ignore abstract, letter
    # take all dialogues, dissertation, essays, political discourses
    # choose 500 random paragraphs from history, treatise, enquiry

    history_dataset = df[df.genre=='history'].sample(n=500, random_state=13)
    treatise_dataset = df[df.genre=='treatise'].sample(n=500, random_state=13)
    enquiry_dataset = df[df.genre=='enquiry'].sample(n=500, random_state=13)

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

    train_test_dataset = unsplit_dataset.train_test_split(test_size=0.2, shuffle=True, seed=13)

    test_eval_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=13)

    dataset = DatasetDict({'train' : train_test_dataset['train'], 
                           'eval' : test_eval_dataset['train'],
                           'test' : test_eval_dataset['test']})
    
    return dataset
