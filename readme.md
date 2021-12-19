# Sibyl-AI
##### Provided under MIT License by Francesco Baldisserri
*Note: this library may be subtly broken or buggy. The code is released under
the MIT License – please take the following message to heart:*
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
> 
## Benefits
- Sibyl is a simple wrapper of SciKit Learn Pipeline
- The default pipeline steps can be used as a simple AutoML tool
- The package includes Keras and Catboost serializable wrappers to be able to use these models with the Pipeline and save them as any other SKLearn estimator
- The default pipeline steps include an OmniEncoder which recognizes the feature types and already transforms them with the most appropriate encoding

## Getting Started
This README is documentation on the syntax of the Sibyl module in this repository. See function docstrings for full syntax details.  
**This module attempts to add features to SKLearn Pipeline module, but in order to use it
to its full potential, you should familiarize yourself with the official SKLearn documentation.**

- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

- You may manually install the project or use ```pip```:
```python
pip install sibyl-ai
#or
pip install git+git://github.com/sirbradflies/Sibyl.git
```

### Sibyl as AutoML
Sibyl framework can be used with its default steps as a simple AutoML pipeline.
The standard steps are:
1. ("omni", OmniEncoder()) (encoder for all features with simple heuristic to recognize continuous, discrete and categorial variables)
2. ("pca", PCA())
3. ("model", KerasDenseRegressor()) or ("model", KerasDenseClassifier()) depending on the predictor pipeline's task

The score used is r2 for the Regressor and accuracy for the Classifier

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sibyl import predictor as pred

X, y = datasets.load_boston(return_X_y=True)  # No encoding needed since OmniEncoder recognizes discrete, continuous and categoricals
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
predrgr = pred.SibylRegressor()
predrgr.search(X_train, y_train)  # RandomizedGridSearchCV available but also fit method is available
print(f"Test score: {predrgr.score(X_test, y_test):.4f}")
```

### Sibyl for custom pipelines
The Sibyl framework can be also used to setup a custom pipeline that includes all features of the SKLearn pipeline but also embeds RandomizedGridSearch and Keras and CatBoost models that can be easily serialized and saved.

Here's a custom pipeline for MNIST digit classification

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sibyl import predictor as pred
from sibyl.models import kerasmodels as kd

STEPS = [("preprocessing", PCA()),
         ("model", kd.KerasDenseClassifier(n_iter_no_change=1,
                                           epochs=1000))]

SEARCH_PARAMS = {"preprocessing__n_components": [None, 0.99, 0.90],
                 "model__units": [(64,), (64, 64), (64, 64, 64)],
                 "model__batch_norm": [True, False]}

X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
predclf = pred.SibylClassifier(steps=STEPS)
predclf.search(X_train, y_train,
               params=SEARCH_PARAMS)  # RandomizedGridSearchCV available but also fit method is available
print(f"Test score: {predclf.score(X_test, y_test):.4f}")
```