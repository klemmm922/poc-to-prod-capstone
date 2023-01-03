import unittest
import pandas as pd
from unittest.mock import MagicMock
import numpy as np

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_train_samples to return the value 80
        base._get_num_train_samples = MagicMock(return_value=80)
        # we assert that _get_num_train_batches will return 80 / batch_size = 4
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        """
        same idea as what we did to test _get_num_test_batches
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_test_samples to return the value 20
        base._get_num_test_samples = MagicMock(return_value=20)
        # we assert that _get_num_test_batches will return 20 / batch_size = 1
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        """
        same idea to test get_label_to_index_map
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_label_list to return the value ['a', 'b', 'c']
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        # we assert that get_index_to_label_map will return {0: 'a', 1: 'b', 2: 'c'}
        self.assertEqual(base.get_index_to_label_map(), {0: 'a', 1: 'b', 2: 'c'})

    def test_index_to_label_and_label_to_index_are_identity(self):
        """
        same idea to test get_index_to_label_map and get_label_to_index_map
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_label_list to return the value ['a', 'b', 'c']
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])

        index_to_label_map = base.get_index_to_label_map()
        lb_to_ind_map  = { label: index for index, label in index_to_label_map.items()}
        #label_to_index_map = base.get_label_to_index_map()
        
        self.assertEqual(base.get_label_to_index_map(), lb_to_ind_map)

        # a completer pas sur du test a faire ici

    def test_to_indexes(self):
        """
        same idea to test to_indexes
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_label_list to return the value ['a', 'b', 'c']
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        # we mock get_label_to_index_map to return the value {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(base.to_indexes(['a', 'b', 'c']), [0, 1, 2])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))

        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)

        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })

        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test_get_num_samples_is_correct(self):
        # TODO: CODE HERE
        """
        same idea as we did for testing BaseTextCategorizationDataset functions, but here it's for
        TestLocalTextCategorizationDataset functions.
        here we are testing the function _get_num_samples
        """
        
        #_dataset = pd.DataFrame(pd.random(100, 5), columns=['post_id', 'tag_name', 'tag_id', 'tag_position', 'title'])

        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3','id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3','title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.6,
                                                       min_samples_per_label=2)
        expected = 6
        print(dataset._get_num_samples())
        self.assertEqual(dataset._get_num_samples(), expected)



    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        """
        Here we are testing the function _get_train_batch
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3','id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3','title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.5,
                                                       min_samples_per_label=1)

        x, y = dataset._get_train_batch()
        print(y.shape)
        self.assertTupleEqual(x.shape, (1,)) and self.assertTupleEqual(y.shape, (1, 5))


    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        """
        Here we are testing the function _get_test_batch_returns_expected_shape
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3','id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3','title_4', 'title_5', 'title_6']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.5,
                                                       min_samples_per_label=1)

        x, y = dataset._get_test_batch()
        print(y.shape)
        self.assertTupleEqual(x.shape, (1,)) and self.assertTupleEqual(y.shape, (1, 5))


    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        """
        Here we are testing the function _get_train_batch_raises_assertion_error
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_a'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset("fake_path", batch_size=3, train_ratio=0.5, min_samples_per_label=1)