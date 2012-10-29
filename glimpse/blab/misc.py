import logging
import numpy as np
import os

from .experiment import ExpModel, ExpCorpus
from glimpse import backends
from glimpse.backends import InsufficientSizeException
from glimpse import pools
from glimpse import util
from glimpse.util import docstring
import glimpse.models

__POOL = None
__MODEL_CLASS = None
__PARAMS = None
__LAYER = None
__EXP = None
__VERBOSE = False

def ConcatStringList(strList):
	result = ''
	for x in strList:
		result += x
	return result
	
def SetPool(pool):
  """Set the worker pool used for this experiment."""
  global __POOL
  if util.IsString(pool):
    pool = pool.lower()
    if pool == 'singlecore':
      pool = pools.SinglecorePool()
    elif pool == 'multicore':
      pool = pools.MulticorePool()
    else:
      raise ValueError("Unknown pool type: %s" % pool)
  logging.info("Using pool type: %s", type(pool).__name__)
  __POOL = pool
  return pool

def GetPool():
  """Get the current worker pool used for new experiments."""
  global __POOL
  if __POOL == None:
    __POOL = pools.MakePool()
  return __POOL

def UseCluster(config_file = None, chunksize = None):
  """Use a cluster of worker nodes for any following experiment commands.

  :param str config_file: path to the cluster configuration file

  """
  pkg = pools.GetClusterPackage()
  pool = pkg.MakePool(config_file, chunksize)
  SetPool(pool)

def SetModelClass(model_class = None):
  """Set the model type.

  :param model_class: Model class to use for future experiments. If a name is
     given, it is passed to :func:`GetModelClass
     <glimpse.models.GetModelClass>`.
  :type model_class: class or str

  """
  if not isinstance(model_class, type):
    model_class = glimpse.models.GetModelClass(model_class)
  logging.info("Using model type: %s", model_class.__name__)
  __MODEL_CLASS = model_class
  return __MODEL_CLASS

def GetModelClass(model_class = None):
  """Get the type that will be used to construct an experiment."""
  if model_class == None:
    __MODEL_CLASS = SetModelClass()
  return __MODEL_CLASS

def SetParams(params = None):
  """Set the parameter object that will be used to construct the next
  experiment.

  :param params: Model-specific parameter object to use for future experiments. If
     a filename is given, the parameter object is read from the given file. If no
     parameter object is given, the model's default parameters are used.
  :type params: object or str

  """
  global __PARAMS
  if params == None:
    params = GetModelClass().ParamClass()
  elif isinstance(params, basestring):
    params = util.Load(params)
  __PARAMS = params
  return __PARAMS

def GetParams():
  """Return the parameter object that will be used to construct the next
  experiment.

  """
  if __PARAMS == None:
    SetParams()
  return __PARAMS

def SetLayer(layer = None):
  """Set the layer from which features will be extracted for the next
  experiment.

  :param layer: Layer from which to compute feature vectors.
  :type layer: str or :class:`glimpse.models.misc.LayerSpec`

  """
  if layer == None:
    layer = GetModelClass().LayerClass.TopLayer()
  elif isinstance(layer, str):
    layer = GetModelClass().LayerClass.FromName(layer)
  __LAYER = layer
  return __LAYER

def GetLayer():
  """Return the layer from which features will be extracted for the next
  experiment.

  """
  return SetLayer()


def GetExperiment():
  """Get the current experiment object.

  :rtype: :class:`Experiment`

  """
  if __EXP == None:
    SetExperiment()
  return __EXP

def SetExperiment(model = None, layer = None):
  """Manually create a new experiment.

  This function generally is not called directly. Instead, an experiment object
  is implicitly created when needed.

  :param model: The Glimpse model to use for processing images.
  :param layer: The layer activity to use for features vectors.
  :type layer: :class:`LayerSpec <glimpse.models.misc.LayerSpec>` or str
  :returns: The new experiment object.
  :rtype: :class:`Experiment`

  .. seealso::
    :func:`SetPool`, :func:`SetParams`, :func:`SetModelClass`, :func:`SetLayer`.

  """
  global __EXP
  if model == None:
    model = MakeModel()
  if layer == None:
    layer = GetLayer()
  elif isinstance(layer, str):
    layer = model.LayerClass.FromName(layer)
  __EXP = Experiment(model, layer, pool = GetPool())
  return __EXP

def MakeModel(model = None, layer = None, pool = None, params = None, prototype_source = None):
  model_class = glimpse.models.GetModelClass()
  if params == None:
    params = model_class.ParamClass()
  if model == None:
    model = model_class(backends.MakeBackend(), params)
  if layer == None:
    layer = SetLayer(layer)
  if pool == None:
    pool = GetPool()
  return model, layer, pool

def MakeDefaultLayer(layer = None):
  return SetLayer(layer)


def SetCorpusSubdirs(corpus_subdirs, corpus = None, classes = None,
    balance = False):
  """Read images from the corpus directory.

  Training and testing subsets are chosen automatically.

  :param corpus_subdirs: Paths to directories for each class. Order
     corresponds to `classes` argument, if set.
  :type corpus_subdirs: list of str
  :param corpus: Path to main corpus directory.
  :type corpus: str, optional
  :type classes: list of str, optional
  :param classes: Set of class names. Use this to ensure a given order to the
     SVM classes. When applying a binary SVM, the first class is treated as
     positive and the second class is treated as negative.
  :type balance: bool, optional
  :param balance:  Ensure an equal number of images from each class (by random
     selection).

  .. seealso::
     :func:`SetTrainTestSplit`

  """
  if classes == None:
    classes = map(os.path.basename, corpus_subdirs)
  try:
    dir_reader = DirReader(ignore_hidden = True)  
    images_per_class = map(dir_reader.ReadFiles, corpus_subdirs)
  except OSError, ex:
    sys.exit("Failed to read corpus directory: %s" % ex)
  # Randomly reorder image lists.
  for images in images_per_class:
    np.random.shuffle(images)
  if balance:
    # Make sure each class has the same number of images.
    num_images = map(len, images_per_class)
    size = min(num_images)
    if not all(n == size for n in num_images):
      images_per_class = [ images[:size] for images in images_per_class ]
  # Use first half of images for training, and second half for testing.
  train_images = [ images[ : len(images)/2 ]
      for images in images_per_class ]
  test_images = [ images[ len(images)/2 : ]
      for images in images_per_class ]
  return ExpCorpus(corpus, train_images, classes), ExpCorpus(corpus, test_images, classes)

class DirReader(object):
  """Read directory contents."""

  def __init__(self, ignore_hidden = True):
    self.ignore_hidden = ignore_hidden

  @staticmethod
  def _HiddenPathFilter(path):
    # Ignore "hidden" entries in directory.
    return not path.startswith('.')

  def _Read(self, dir_path):
    entries = os.listdir(dir_path)
    if self.ignore_hidden:
      entries = filter(DirReader._HiddenPathFilter, entries)
    return [ os.path.join(dir_path, entry) for entry in entries ]

  def ReadDirs(self, dir_path):
    """Read list of sub-directories."""
    return filter(os.path.isdir, self._Read(dir_path))

  def ReadFiles(self, dir_path):
    """Read list of files."""
    return filter(os.path.isfile, self._Read(dir_path))

    """Read images from the corpus directory.

    Training and testing subsets are chosen automatically.

    :param str corpus_dir: Path to corpus directory.
    :type classes: list of str
    :param classes: Set of class names. Use this to ensure a given order to the
       SVM classes. When applying a binary SVM, the first class is treated as
       positive and the second class is treated as negative.
    :param bool balance: Ensure an equal number of images from each class (by
       random selection).

    .. seealso::
       :func:`SetTrainTestSplit`

    """
    if classes == None:
      corpus_subdirs = self.dir_reader.ReadDirs(corpus_dir)
    else:
      corpus_subdirs = [ os.path.join(corpus_dir, cls) for cls in classes ]
      # Check that sub-directory exists.
      for subdir in corpus_subdirs:
        if not os.path.isdir(subdir):
          raise ValueError("Directory not found: %s" % subdir)
    return self.SetCorpusSubdirs(corpus_subdirs, corpus_dir, classes, balance)

def SetCorpus(corpus_dir, classes = None, balance = False):
  """Read images from the corpus directory.

  Training and testing subsets are chosen automatically.

  :param str corpus_dir: Path to corpus directory.
  :type classes: list of str
  :param classes: Set of class names. Use this to ensure a given order to the
     SVM classes. When applying a binary SVM, the first class is treated as
     positive and the second class is treated as negative.
  :param bool balance: Ensure an equal number of images from each class (by
     random selection).

  .. seealso::
     :func:`SetTrainTestSplit`

  """
  dir_reader = DirReader(ignore_hidden = True)  
  
  if classes == None:
    corpus_subdirs = dir_reader.ReadDirs(corpus_dir)
  else:
    corpus_subdirs = [ os.path.join(corpus_dir, cls) for cls in classes ]
    # Check that sub-directory exists.
    for subdir in corpus_subdirs:
      if not os.path.isdir(subdir):
        raise ValueError("Directory not found: %s" % subdir)
  train, test = SetCorpusSubdirs(corpus_subdirs, corpus_dir, classes, balance)
  train.name = corpus_dir + "Train"
  test.name = corpus_dir + "Test"
  return train, test

