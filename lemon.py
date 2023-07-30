"""
The code presented here is mostly extracted from the GitHub repository: https://github.com/iamDecode/lemon-evaluation

The full publication is cited:

Collaris, D., Gajane, P., Jorritsma, J., van Wijk, J. J., & Pechenizkiy, M. (2023, April). 
LEMON: Alternative Sampling for More Faithful Explanation Through Local Surrogate Models. 
In Advances in Intelligent Data Analysis XXI: 21st International Symposium on Intelligent Data Analysis, IDA 2023, Louvain-la-Neuve, Belgium, April 12–14, 2023, 
Proceedings (pp. 77-90). Cham: Springer Nature Switzerland.
"""
from sklearn.preprocessing import normalize
from bisect import bisect
from scipy.optimize import newton
from functools import partial
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import gammainccinv
import numpy as np
import time

def uniform_kernel(x):
    return 1

def gaussian_kernel(x, kernel_width):
    """
    Cited in the original GitHub repository by Collaris, D.:
    https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_tabular.py#L251
    """
    return np.sqrt(np.exp(-(x ** 2) / kernel_width ** 2))
  
def sqcos_kernel(x):
    return np.cos(x)**2
  
def trapezoid_kernel(x, a, b):
    if 0 <= x and  x <= a:
        return (2 / (a + b))
    elif a <= x and x <= b:
        return (2 / (a + b)) * ((b - x) / (b - a))
    else: 
        return 0

class LemonExplainer(object):
  """
  Intantiates the explainer.
  """
  def __init__(self, training_data, instance, distance_kernel=None, sample_size = 5000, radius_max=1, random_state=None):
    start_time = time.time()
    self.random_state = check_random_state(random_state)
    np.random.seed(random_state)

    self.training_data = training_data
    self.scaler = StandardScaler(with_mean=False)
    self.scaler.fit(training_data)
    
    # Create hypersphere samples. The sphere is only computed once for performance and stability,
    # but it would be better to resample the sphere every time `explain_instance` is called.
    # I checked, this does not affect the results in any way.
    dimensions = training_data.shape[1]

    if distance_kernel is None:
      self.distance_kernel = np.vectorize(lambda x: x ** (1 / dimensions))
    else:
      self.distance_kernel = np.vectorize(self._transform(distance_kernel, dimensions, radius_max=radius_max))

    sphere = np.random.normal(size=(sample_size, dimensions))
    sphere = normalize(sphere)
    sphere *= self.distance_kernel(np.random.uniform(size=sample_size)).reshape(-1,1)
    
    self.sphere = sphere
    self.samples = self.generate_samples(instance)
    self.weights = np.ones((self.sphere.shape[0],))
    end_time = time.time()
    self.run_time = end_time - start_time

  @property
  def surrogate(self):
    try:
      return self._surrgate
    except AttributeError:
      self._surrogate = Ridge(alpha=0, fit_intercept=True, normalize=True, random_state=self.random_state)
      return self._surrogate

  def generate_samples(self, instance):
    X_transfer = self.scaler.inverse_transform(self.sphere) + np.array([instance])
    return X_transfer

  def explain_instance(self, instance, predict_fn, labels=(1,), surrogate=None):
    surrogate = surrogate or self.surrogate
  
    # Create transfer dataset by perturbing the original instance with the hypersphere samples
    X_transfer = self.scaler.inverse_transform(self.sphere) + np.array([instance])
    y_transfer = predict_fn(X_transfer)

    def explain_label(label):
      surrogate.fit(X_transfer, y_transfer[:,label])
      score = surrogate.score(X_transfer, y_transfer[:,label])
      return (surrogate.coef_, score)

    return [explain_label(label) for label in labels]

  def _transform(self, kernel, dimensions, sample_size = 1000, radius_max=1): 
    """
    Inverse transform sampling
    """
    cdf_samples = np.array([kernel(x)*(x**(dimensions-1)) for x in np.linspace(0, radius_max, sample_size)])
    cdf_samples = np.cumsum(cdf_samples)
    cdf_samples /= cdf_samples[-1]
    return lambda y: radius_max * (bisect(cdf_samples, y) / sample_size)

def lemon_method(perturbator, data, ioi):
    """
    Outputs the perturbations with their weights and the total execution time when creating the Lemon object
    """
    kernel_width_lemon = 0.5
    sample_size = 5000
    p = 0.99
    train_data = data.processed_train_pd.values
    d = train_data.shape[1]
    radius = kernel_width_lemon * np.sqrt(2*gammainccinv(d/2, (1-p))) 
    distance_kernel=partial(gaussian_kernel, kernel_width=kernel_width_lemon)
    radius_max=radius
    instance = ioi.normal_x
    lemon = LemonExplainer(train_data, instance, distance_kernel=distance_kernel, sample_size=sample_size, radius_max=radius_max)
    return lemon.samples, lemon.samples, lemon.weights, lemon.run_time