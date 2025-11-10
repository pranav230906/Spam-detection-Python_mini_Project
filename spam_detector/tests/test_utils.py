# Placeholder for test_utils.py
# tests/test_utils.py
import os
from utils import load_dataset
import pandas as pd

def test_load_csv(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("label,text\nspam,Buy now\nham,hello friend\n")
    df = load_dataset(str(p))
    assert len(df) == 2
    assert set(df['label']) == {'spam','ham'}

def test_load_raw(tmp_path):
    p = tmp_path / "raw.txt"
    p.write_text("spam\tBuy now\nham\thello friend\n")
    df = load_dataset(str(p))
    assert len(df) == 2
