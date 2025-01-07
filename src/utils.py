import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def create_folds(input_file_path, output_file_path, num_folds = 5):
    df = pd.read_csv(input_file_path)
    df["FOLD"] = -1
    target_cols = ["cohesion","syntax","vocabulary","phraseology","grammar","conventions"]
    skf = MultilabelStratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 42)
    for i, (train_idx, val_idx) in enumerate(skf.split(df, df[target_cols])):
        df.loc[val_idx, "FOLD"] = i+1

    df.to_csv(output_file_path, index = False)