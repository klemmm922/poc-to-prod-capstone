from predict.predict import run
from train.train import run as train_run
from preprocessing.preprocessing import utils

import unittest
from unittest.mock import MagicMock
import tempfile

from predict.predict.run import TextPredictionModel

import pandas as pd

def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })

class TestPred(unittest.TestCase):

    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):

        params = {
            "batch_size": 2,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 4,
            "verbose": 1
        }

        text_list = ["Is it possible to execute the procedure of a function in the scope of the caller?"]
    
        model = TextPredictionModel.from_artefacts("train/data/artefacts/")
        predictions = model.predict(text_list,top_k=1)

        self.assertEqual(predictions, [['php']])