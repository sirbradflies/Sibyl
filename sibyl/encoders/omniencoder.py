"""
Represent a type of data in a ML dataset
@author: Francesco Baldisserri
@creation date: 24/06/2020
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

INTEGERS = ['int8', 'int16', 'int32', 'int64']
DECIMALS = ['float16', 'float32', 'float64']
NUMBERS = INTEGERS+DECIMALS

NUM_TYPE, CAT_TYPE, TXT_TYPE, IGNORE = "number", "categorical", "text", "IGNORE"


class OmniEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_encoder=None, cat_encoder=None, max_words=100,
                 sparse_threshold=0.3):
        self.num_encoder = num_encoder
        self.cat_encoder = cat_encoder
        self.encoder = None
        self.max_words = max_words
        self.sparse_threshold = sparse_threshold

    def fit(self, X, y=None):
        self.encoder = self.build_encoder(X)
        return self.encoder.fit(X)

    def transform(self, X, y=None):
        return self.encoder.transform(X)

    def build_encoder(self, values):
        """ Build an encoder based on the types of values passed """
        types = get_types(values)
        num_transf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler() if self.num_encoder is None
            else self.num_encoder)
        cat_transf = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore") if self.cat_encoder is None
            else self.cat_encoder)
        txt_transf = ColumnTransformer(
            [(TXT_TYPE + "_" + str(x),
              TfidfVectorizer(max_features=self.max_words), x)
             for x in types[TXT_TYPE]]
        )
        return ColumnTransformer(transformers=[
            (NUM_TYPE, num_transf, types[NUM_TYPE]),
            (CAT_TYPE, cat_transf, types[CAT_TYPE]),
            (TXT_TYPE, txt_transf, types[TXT_TYPE])
        ], sparse_threshold=self.sparse_threshold)


def get_type(value_column):
    """Classify the value of a field by DataType class"""
    values = pd.Series(value_column)
    temp_values = pd.to_numeric(values.dropna(),
                                errors="ignore",
                                downcast="integer")
    total_values, unique_values = len(temp_values), len(temp_values.unique())
    if unique_values <= 1:  # ignore fields with 0 or values
        data_type = IGNORE
    elif temp_values.dtype in NUMBERS:  # Number type
        data_type = NUM_TYPE
    elif unique_values < total_values/10:  # If unique values less than 10% of total values then it's a categorical
        data_type = CAT_TYPE
    elif temp_values.apply(type).eq(str).all() and \
            temp_values.apply(len).median() > 50 and \
            not values.isnull().any():  # String columns with median len>50
        data_type = TXT_TYPE
    else:  # Anything else is ignored
        data_type = IGNORE
    return data_type


def get_types(value_matrix):
    """ Classify multi-column dataset by data type
    and return a dictionary with type:[features]"""
    types = {NUM_TYPE: [], CAT_TYPE: [], TXT_TYPE: [], IGNORE: []}
    data = pd.DataFrame(value_matrix)
    for column in data.columns:
        column_type = get_type(data[column])
        types[column_type] += [column]
    return types
