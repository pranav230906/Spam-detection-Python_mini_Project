# Placeholder for utils.py
# utils.py
import pandas as pd
import os
import re

def load_dataset(path):
    """
    Load CSV (with header), TSV, or UCI SMSSpamCollection (tab separated no header).
    Normalize to DataFrame with columns ['label','text'] where label is 'spam' or 'ham'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    _, ext = os.path.splitext(path.lower())
    if ext in (".csv",):
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='warn')
        # common column names mapping
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1':'label','v2':'text'})
        if 'label' in df.columns and 'text' in df.columns:
            df = df[['label','text']]
        elif 'target' in df.columns and 'message' in df.columns:
            df = df.rename(columns={'target':'label','message':'text'})[['label','text']]
        else:
            # try to find first two columns as label,text
            cols = list(df.columns)
            if len(cols) >= 2:
                df = df[[cols[0], cols[1]]].rename(columns={cols[0]:'label', cols[1]:'text'})
            else:
                raise ValueError("CSV format not recognized. Expect columns like 'label' and 'text'.")
    else:
        # try tab-separated (UCI SMSSpamCollection)
        df = pd.read_csv(path, sep='\t', header=None, names=['label','text'], encoding='utf-8', on_bad_lines='warn')

    # normalize labels
    df['label'] = df['label'].astype(str).str.strip().str.lower().map(lambda x: 'spam' if x in ('spam','1','true','t') else 'ham')
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    df = df.reset_index(drop=True)
    return df
