"""
Generic regression model for machine learning problems based on Keras

@author: Francesco Baldisserri
@creation date: 06/03/2020
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import scipy
import shutil
import random
import tempfile
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as kl
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


# TODO: Refactor with Estimator template:
# https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
class KerasDenseRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, units=(64, 64), dropout=0, activation="relu",
                 batch_norm=False, batch_size=None, optimizer="nadam",
                 kernel_init="glorot_normal", val_split=0.2, epochs=10,
                 loss="mean_squared_error", n_iter_no_change=None,
                 custom_objects=None, _estimator_type="regressor"):
        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.kernel_init = kernel_init
        self.val_split = val_split
        self.epochs = epochs
        self.loss = loss
        self.n_iter_no_change = n_iter_no_change
        self.model = None
        self.custom_objects = custom_objects
        self._estimator_type = _estimator_type

    def fit(self, X, y, out_act="linear", loss="mse", metrics=None):
        X_val, y_val = _validate_array(X), _validate_array(y)
        if self.model is None:
            self._build_model(X_val, y_val, out_act=out_act,
                              loss=loss, metrics=metrics)
        calls = [] if self.n_iter_no_change is None else\
            [EarlyStopping(monitor="val_loss" if self.val_split > 0 else "loss",
                           patience=self.n_iter_no_change,
                           restore_best_weights=True)]
        try:
            return self.model.fit(X_val, y_val, epochs=self.epochs, verbose=0,
                                  callbacks=calls, validation_split=self.val_split)
        except Exception as e:
            return None

    def predict(self, X):
        return self.model.predict(_validate_array(X))

    def _build_model(self, X, y, out_act, loss, metrics):
        """ Build Keras model according to input parameters """
        in_shape = X.shape[1] if len(X.shape) > 1 else 1
        layers = [kl.InputLayer(in_shape)]
        #layers = [kl.Flatten()] if len(X.shape[1:]) > 1 else []
        for units in self.units:
            if self.dropout > 0: layers += [kl.Dropout(self.dropout)]
            layers += [kl.Dense(units, activation=self.activation,
                                kernel_initializer=self.kernel_init)]
            if self.batch_norm: layers += [kl.BatchNormalization()]
        out_shape = y.shape[1] if len(y.shape) > 1 else 1
        layers += [kl.Dense(out_shape, activation=out_act)]
        self.model = keras.models.Sequential(layers)
        self.model.compile(loss=loss, metrics=metrics,
                           optimizer=self.optimizer)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            keras.models.save_model(self.model, temp.name)
            with temp:
                state["model"] = temp.read()
            os.remove(temp.name)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if state["model"] is not None:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            with temp:
                temp.write(state["model"])
            self.model = keras.models.load_model(temp.name,
                                                 custom_objects=self.custom_objects)
            os.remove(temp.name)

    """def __getstate__(self):
        tmp_path = os.path.join(tempfile.gettempdir(), "sibyl_temp")
        state = self.__dict__.copy()
        if self.model is not None:
            keras.models.save_model(self.model, tmp_path, save_format="tf")
            shutil.make_archive(tmp_path, "zip", tmp_path)
            shutil.rmtree(tmp_path)
            with open(f"{tmp_path}.zip", mode="rb") as temp:
                state["model"] = temp.read()
            os.remove(f"{tmp_path}.zip")
        return state

    def __setstate__(self, state):
        tmp_path = os.path.join(tempfile.gettempdir(), "sibyl_temp")
        self.__dict__ = state
        if state["model"] is not None:
            with open(f"{tmp_path}.zip", mode="wb") as temp:
                temp.write(state["model"])
                temp.flush()
            shutil.unpack_archive(f"{tmp_path}.zip", tmp_path)
            os.remove(f"{tmp_path}.zip")
            self.model = keras.models.load_model(tmp_path,
                                                 custom_objects=self.custom_objects)
            shutil.rmtree(tmp_path)"""


class KerasDenseClassifier(KerasDenseRegressor, ClassifierMixin):
    def __init__(self, units=(64, 64), dropout=0, activation="relu",
                 batch_norm=False, batch_size=None, optimizer="nadam",
                 kernel_init="glorot_normal", val_split=0.2, epochs=10,
                 loss="categorical_crossentropy", n_iter_no_change=None,
                 custom_objects=None, _estimator_type="classifier"):
        self.encoder = LabelBinarizer()
        super().__init__(units=units, dropout=dropout,
                         activation=activation, batch_norm=batch_norm,
                         batch_size=batch_size,  optimizer=optimizer,
                         kernel_init=kernel_init, val_split=val_split,
                         epochs=epochs, loss=loss,
                         n_iter_no_change=n_iter_no_change,
                         custom_objects=custom_objects)

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        columns = y_enc.shape[1]
        loss, out_act = ("binary_crossentropy", "sigmoid") if columns==1 \
            else ("categorical_crossentropy", "softmax")
        return super().fit(X, y_enc, out_act=out_act,
                           loss=loss, metrics=["accuracy"])

    def predict(self, X):
        y_enc = super().predict(X)
        return self.encoder.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class KerasCNNRegressor(KerasDenseRegressor):
    def __init__(self, cnn_units=(64, 64), dense_units=(64,64), dropout=0,
                 activation="relu", batch_norm=False, batch_size=None,
                 optimizer="nadam", kernel_init="glorot_normal", val_split=0.2,
                 epochs=10, loss="mean_squared_error", n_iter_no_change=None,
                 custom_objects=None, _estimator_type="regressor"):
        self.cnn_units = cnn_units
        self.dense_units = dense_units
        super().__init__(dropout = dropout,
                         activation = activation,
                         batch_norm = batch_norm,
                         batch_size = batch_size,
                         optimizer = optimizer,
                         kernel_init = kernel_init,
                         val_split = val_split,
                         epochs = epochs,
                         loss = loss,
                         n_iter_no_change = n_iter_no_change,
                         custom_objects = custom_objects,
                         _estimator_type = _estimator_type)

    def _build_model(self, X, y, out_act, loss, metrics):
        """ Build Keras model according to input parameters """
        layers = [kl.Reshape(X.shape[1:]+(1,))] if len(X.shape[1:]) < 2 else []
        for units in self.cnn_units:
            if self.dropout > 0: layers += [kl.Dropout(self.dropout)]
            layers += [kl.Conv1D(filters=units, kernel_size=2,
                                 dilation_rate=2, padding="causal",
                                 activation=self.activation)]
            if self.batch_norm: layers += [kl.BatchNormalization()]
        layers += [kl.Flatten()]
        for units in self.dense_units:
            if self.dropout > 0: layers += [kl.Dropout(self.dropout)]
            layers += [kl.Dense(units, activation=self.activation,
                                kernel_initializer=self.kernel_init)]
            if self.batch_norm: layers += [kl.BatchNormalization()]
        out_shape = y.shape[1] if len(y.shape) > 1 else 1
        layers += [kl.Dense(out_shape, activation=out_act)]
        self.model = keras.models.Sequential(layers)
        self.model.compile(loss=loss, metrics=metrics,
                           optimizer=self.optimizer)


class KerasCNNClassifier(KerasCNNRegressor):
    def __init__(self, cnn_units=(64, 64), dense_units=(64, 64), dropout=0,
                 activation="relu", batch_norm=False, batch_size=None,
                 optimizer="nadam", kernel_init="glorot_normal", val_split=0.2,
                 epochs=10, loss="categorical_crossentropy",
                 n_iter_no_change=None, custom_objects=None,
                 _estimator_type="classifier"):
        self.encoder = LabelBinarizer()
        super().__init__(
            cnn_units = cnn_units,
            dense_units= dense_units,
            dropout = dropout,
            activation = activation,
            batch_norm = batch_norm,
            batch_size = batch_size,
            optimizer = optimizer,
            kernel_init = kernel_init,
            val_split = val_split,
            epochs = epochs,
            loss = loss,
            n_iter_no_change = n_iter_no_change,
            custom_objects = custom_objects,
            _estimator_type = _estimator_type)

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        columns = y_enc.shape[1]
        loss, out_act = ("binary_crossentropy", "sigmoid") if columns == 1 \
            else ("categorical_crossentropy", "softmax")
        return super().fit(X, y_enc, out_act=out_act,
                           loss=loss, metrics=["accuracy"])

    def predict(self, X):
        y_enc = super().predict(X)
        return self.encoder.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _validate_array(array):
    if isinstance(array, list):
        validated_array = np.array(array)
    elif scipy.sparse.issparse(array):
        validated_array = array.todense()
    else:
        validated_array = array
    return validated_array
