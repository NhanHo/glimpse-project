#!/usr/bin/python

"""The glab module provides a command-driven, matlab-like interface to the
Glimpse project.

The following is an example of a basic experiment.

from glimpse.glab import *

# Read images from the directory "indir/corpus", which is assumed to have two
# sub-directories (one for each class). The names of the sub-directories
# corresponds to the class names. The first sub-directory (as given by the
# os.listdir() method) is the "positive" class, while the second is the
# "negative" class. Half of the images from each class are randomly selected to
# form the training set, while the other half is held for testing.
SetCorpus("indir/corpus")
# Imprint a set of 10 S2 prototypes from the training set.
ImprintS2Prototypes(10)
# Build and test an SVM classifier based on IT activity.
RunSvm()
# Store the configuration and results of the experiment to the file
# "outdir/exp.dat". Additionally, store the SVM classifier to the file
# "outdir/exp.dat.svm".
StoreExperiment("outdir/exp.dat")

# By default, the order of the classes is somewhat arbitrary. Instead, we could
# specify that "class1" is the positive SVM class, while "class2" is the
# negative SVM class,
SetCorpus('corpus', classes = ('cls1', 'cls2'))

# If we had wanted to build SVM feature vectors from C1 activity, instead of IT
# activity, we could have initialized the experiment before setting the corpus.
SetExperiment(layer = 'C1')
SetCorpus("indir/corpus")

# If we had wanted to configure the parameters of the Glimpse model, we could
# have constructed the model manually when initializing the experiment.
params = Params()
params.num_scales = 8
SetExperiment(model = Model(params))
SetCorpus("indir/corpus")
"""

from glimpse import backends
from glimpse.models import viz2
from glimpse import pools
from glimpse import util
from glimpse.util.svm import SpheringFeatureScaler, PrepareLibSvmInput, \
    SvmForSplit, SvmCrossValidate
import itertools
import logging
import numpy as np
import operator
import os
import sys
import time

__all__ = ( 'SetPool', 'UseCluster', 'SetModelClass', 'MakeParams', 'MakeModel',
    'GetExperiment', 'SetExperiment', 'ImprintS2Prototypes',
    'MakeRandomS2Prototypes', 'SetS2Prototypes', 'SetCorpus',
    'SetTrainTestSplit', 'SetTrainTestSplitFromDirs', 'ComputeFeatures',
    'RunSvm', 'LoadExperiment', 'StoreExperiment', 'Verbose')

def SplitList(data, *sizes):
  """Break a list into sublists.
  data -- (list) input data
  sizes -- (int list) size of each chunk. if sum of sizes is less than entire
           size of input array, the remaining elements are returned as an extra
           sublist in the result.
  RETURN (list of lists) sublists of requested size
  """
  assert(all([ s >= 0 for s in sizes ]))
  if len(sizes) == 0:
    return data
  if sum(sizes) < len(data):
    sizes = list(sizes)
    sizes.append(len(data) - sum(sizes))
  out = list()
  last = 0
  for s in sizes:
    out.append(data[last : last+s])
    last += s
  return out

class Experiment(object):

  def __init__(self, model, layer, pool, scaler):
    """Create a new experiment.
    model -- the Glimpse model to use for processing images
    layer -- (LayerSpec) the layer activity to use for features vectors
    pool -- a serializable worker pool
    scaler -- feature scaling algorithm
    """
    # Default arguments should be chosen in SetExperiment()
    assert model != None
    assert layer != None
    assert pool != None
    assert scaler != None
    self.model = model
    self.pool = pool
    self.layer = layer
    self.scaler = scaler
    # Initialize attributes used by an experiment
    self.classes = []
    self.classifier = None
    self.corpus = None
    self.prototype_source = None
    self.train_images = None
    self.test_images = None
    self.train_test_split = None
    self.train_features = None  # (list of 2D array) indexed by class, image,
                                # and then feature offset
    self.test_features = None  # (list of 2D array) indexed as in train_features
    self.train_accuracy = None
    self.test_accuracy = None
    self.cross_validated = None  # (bool) indicates whether cross-validation was
                                 # used to compute test accuracy.
    self.prototype_construction_time = None
    self.svm_train_time = None
    self.svm_test_time = None
    self.debug = False

  @property
  def features(self):
    """The full set of features for each class, without training/testing splits.
    RETURN (list of 2D float ndarray) indexed by class, image, and then feature
    offset.
    """
    if self.train_features == None:
      return None
    # Reorder instances from (set, class) indexing, to (class, set) indexing.
    features = zip(self.train_features, self.test_features)
    # Concatenate instances for each class (across sets)
    features = map(np.vstack, features)
    return features

  @property
  def images(self):
    """The full set of images, without training/testing splits.
    RETURN (list of string lists) indexed by class, and then image.
    """
    if self.train_images == None:
      return None
    # Combine images by class, and concatenate lists.
    return map(util.UngroupLists, zip(self.train_images, self.test_images))

  @property
  def s2_prototypes(self):
    return self.model.s2_kernels

  @s2_prototypes.setter
  def s2_prototypes(self, value):
    self.prototype_source = 'manual'
    self.model.s2_kernels = value

  def __str__(self):
    values = dict(self.__dict__)
    values['classes'] = ", ".join(values['classes'])
    return """Experiment:
  corpus: %(corpus)s
  classes: %(classes)s
  train_test_split: %(train_test_split)s
  model: %(model)s
  layer: %(layer)s
  prototype_source: %(prototype_source)s
  train_accuracy: %(train_accuracy)s
  test_accuracy: %(test_accuracy)s""" % values

  __repr__ = __str__

  def ImprintS2Prototypes(self, num_prototypes):
    """Imprint a set of S2 prototypes from a set of training images.
    num_prototypes -- (int) the number of C1 patches to sample
    """
    if self.train_images == None:
      sys.exit("Please specify the training corpus before imprinting "
          "prototypes.")
    start_time = time.time()
    image_files = util.UngroupLists(self.train_images)
    # Represent each image file as an empty model state.
    input_states = map(self.model.MakeStateFromFilename, image_files)
    prototypes, locations = self.model.ImprintS2Prototypes(num_prototypes,
        input_states, normalize = True, pool = self.pool)
    # Store new prototypes in model.
    self.prototype_source = 'imprinted'
    if self.debug:
      # Convert input source index to corresponding image path.
      locations = [ (image_files[l[0]],) + l[1:] for l in locations ]
      self.debug_prototype_locations = locations
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time

  def MakeRandomS2Prototypes(self, num_prototypes):
    """Create a set of S2 prototypes with uniformly random entries.
    num_prototypes -- (int) the number of S2 prototype arrays to create
    """
    start_time = time.time()
    shape = (num_prototypes,) + tuple(self.model.s2_kernel_shape)
    prototypes = np.random.uniform(0, 1, shape)
    for p in prototypes:
      p /= np.linalg.norm(p)
    self.prototype_source = 'random'
    self.model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time

  def SetS2Prototypes(self, prototypes):
    """Set the S2 prototypes from an array.
    prototypes -- (ndarray) the set of prototypes
    """
    self.prototype_source = 'manual'
    self.model.s2_kernels = prototypes

  def ComputeFeaturesFromInputStates(self, input_states):
    """Return the activity of the model's output layer for a set of images.
    input_states -- (State iterable) model states containing image data
    RETURN (iterable) a feature vector for each image
    """
    L = self.model.Layer
    if self.layer in (L.S2, L.C2, L.IT) and self.model.s2_kernels == None:
      sys.exit("Please set the S2 prototypes before computing feature vectors "
          "for layer %s." % self.layer.name)
    builder = self.model.BuildLayerCallback(self.layer, save_all = False)
    # Compute model states containing IT features.
    output_states = self.pool.map(builder, input_states)
    if self.debug:
      self.debug_output_states = output_states
    # Look up the activity values for the output layer, and convert them all to
    # a single vector.
    return [ util.ArrayListToVector(state[self.layer.id])
        for state in output_states ]

  def _ReadCorpusDir(self, corpus_dir, classes = None):
    if classes == None:
      classes = os.listdir(corpus_dir)
    try:
      def ReadClass(cls):
        class_dir = os.path.join(corpus_dir, cls)
        return [ os.path.join(class_dir, img) for img in os.listdir(class_dir) ]
      return map(ReadClass, classes)
    except OSError, e:
      sys.exit("Failed to read corpus directory: %s" % e)

  def SetCorpus(self, corpus_dir, classes = None):
    """Read images from the corpus directory, and choose training and testing
    subsets automatically. Use this instead of SetTrainTestSplit().
    corpus_dir -- (str) path to corpus directory
    classes -- (list) set of class names. Use this to ensure a given order to
               the SVM classes. When applying a binary SVM, the first class is
               treated as positive and the second class is treated as negative.
    """
    if classes == None:
      classes = os.listdir(corpus_dir)
    self.classes = classes
    self.corpus = corpus_dir
    self.train_test_split = 'automatic'
    images_per_class = self._ReadCorpusDir(corpus_dir, classes)
    # Randomly reorder image lists.
    for images in images_per_class:
      np.random.shuffle(images)
    # Use first half of images for training, and second half for testing.
    self.train_images = [ images[ : len(images)/2 ]
        for images in images_per_class ]
    self.test_images = [ images[ len(images)/2 : ]
        for images in images_per_class ]

  def SetTrainTestSplitFromDirs(self, train_dir, test_dir, classes = None):
    """Read images from the corpus directories, setting the training and testing
    subsets manually. Use this instead of SetCorpus().
    train_dir -- (str) path to directory of training images
    test_dir -- (str) path to directory of testing images
    classes -- (list) class names
    """
    if classes == None:
      classes = os.listdir(train_dir)
    train_images = self._ReadCorpusDir(train_dir, classes)
    test_images = self._ReadCorpusDir(test_dir, classes)
    self.SetTrainTestSplit(train_images, test_images, classes)
    self.corpus = (train_dir, test_dir)

  def SetTrainTestSplit(self, train_images, test_images, classes):
    """Manually specify the training and testing images.
    train_images -- (list of str list) paths for each training image, with one
                    sub-list per class
    test_images -- (list of str list) paths for each training image, with one
                   sub-list per class
    classes -- (list) class names
    """
    self.classes = classes
    self.train_test_split = 'manual'
    self.train_images = train_images
    self.test_images = test_images

  def ComputeFeatures(self):
    """Compute SVM feature vectors for all images."""
    if self.train_images == None or self.test_images == None:
      sys.exit("Please specify the corpus.")
    train_sizes = map(len, self.train_images)
    train_size = sum(train_sizes)
    test_sizes = map(len, self.test_images)
    train_images = util.UngroupLists(self.train_images)
    test_images = util.UngroupLists(self.test_images)
    images = train_images + test_images
    # Compute features for all images.
    input_states = map(self.model.MakeStateFromFilename, images)
    start_time = time.time()
    features = self.ComputeFeaturesFromInputStates(input_states)
    self.compute_feature_time = time.time() - start_time
    # Split results by training/testing set
    train_features, test_features = SplitList(features, train_size)
    # Split training set by class
    train_features = SplitList(train_features, *train_sizes)
    # Split testing set by class
    test_features = SplitList(test_features, *test_sizes)
    # Store features as list of 2D arrays
    self.train_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in train_features ]
    self.test_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in test_features ]

  def RunSvm(self, cross_validate = False):
    """Train and test an SVM classifier from the set of training images.
    cross_validate -- (bool) if true, perform 10x10-way cross-validation.
                      Otherwise, compute accuracy for existing training/testing
                      split.
    RETURN (float tuple) training and testing accuracies (training accuracy is
    None when cross-validating.)
    """
    if self.train_features == None:
      self.ComputeFeatures()
    start_time = time.time()
    if cross_validate:
      self.test_accuracy = SvmCrossValidate(self.features, num_repetitions = 10,
          num_splits = 10, scaler = self.scaler)
      self.train_accuracy = None
    else:
      self.classifier, self.train_accuracy, self.test_accuracy = \
          SvmForSplit(self.train_features, self.test_features,
              scaler = self.scaler)
    self.cross_validated = cross_validate
    self.svm_time = time.time() - start_time
    return self.train_accuracy, self.test_accuracy

  def Store(self, root_path):
    """Save the experiment to disk."""
    # We modify the value of the "classifier" attribute, so cache it.
    classifier = self.classifier
    pool = self.pool
    self.pool = None  # can't serialize some pools
    # Use "classifier" attribute to indicate whether LIBSVM classifier is
    # present.
    if classifier != None:
      self.classifier = True
    util.Store(self, root_path)
    if classifier != None:
      # Use delayed import of LIBSVM library, so non-SVM methods are always
      # available.
      import svmutil
      svmutil.svm_save_model(root_path + '.svm', classifier)
    if self.debug and hasattr(pool, 'cluster_stats'):
      # This is a hackish way to record basic cluster usage information.
      util.Store(pool.cluster_stats, root_path + '.cluster-stats')
    # Restore the value of the "classifier" and "pool" attributes.
    self.classifier = classifier
    self.pool = pool

  @staticmethod
  def Load(root_path):
    """Load the experiment from disk."""
    experiment = util.Load(root_path)
    if experiment.classifier != None:
      # Use delayed import of LIBSVM library, so non-SVM methods are always
      # available.
      import svmutil
      experiment.classifier = svmutil.svm_load_model(root_path + '.svm')
    return experiment

__POOL = None
__MODEL_CLASS = viz2.Model
__EXP = None
__VERBOSE = False

def SetPool(pool):
  """Set the worker pool used for this experiment."""
  global __POOL
  logging.info("Using pool type: %s" % type(pool).__name__)
  __POOL = pool

def UseCluster(config_file = None, chunksize = None):
  """Use a cluster of worker nodes for any following experiment commands.
  config_file -- (str) path to the cluster configuration file
  """
  from glimpse.pools.cluster import ClusterConfig, ClusterPool
  if config_file == None:
    if 'GLIMPSE_CLUSTER_CONFIG' not in os.environ:
      raise ValueError("Please specify a cluster configuration file.")
    config_file = os.environ['GLIMPSE_CLUSTER_CONFIG']
  config = ClusterConfig(config_file)
  SetPool(ClusterPool(config, chunksize = chunksize))

def SetModelClass(model_class):
  """Set the model type.
  model_class -- for example, use glimpse.models.viz2.model.Model
  """
  global __MODEL_CLASS
  logging.info("Using model type: %s" % model_class.__name__)
  __MODEL_CLASS = model_class

def MakeParams():
  """Create a default set of parameters for the current model type."""
  global __MODEL_CLASS
  return __MODEL_CLASS.Params()

def MakeModel(params = None):
  """Create the default model."""
  global __MODEL_CLASS
  if params == None:
    params = MakeParams()
  return __MODEL_CLASS(backends.MakeBackend(), params)

def GetExperiment():
  """Get the current experiment object."""
  global __EXP
  if __EXP == None:
    SetExperiment()
  return __EXP

def SetExperiment(model = None, layer = None, scaler = None):
  """Create a new experiment.
  model -- the Glimpse model to use for processing images
  layer -- (LayerSpec or str) the layer activity to use for features vectors
  scaler -- feature scaling algorithm
  """
  global __EXP, __POOL
  if __POOL == None:
    __POOL = pools.MakePool()
  if model == None:
    model = MakeModel()
  if layer == None:
    layer = model.Layer.IT
  elif isinstance(layer, str):
    layer = model.Layer.FromName(layer)
  if scaler == None:
    scaler = SpheringFeatureScaler()
  __EXP = Experiment(model, layer, pool = __POOL, scaler = scaler)

def ImprintS2Prototypes(num_prototypes):
  """Imprint a set of S2 prototypes from a set of training images.
  num_prototypes -- (int) the number of C1 patches to sample
  """
  if __VERBOSE:
    print "Imprinting %d prototypes" % num_prototypes
  result = GetExperiment().ImprintS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def MakeRandomS2Prototypes(num_prototypes):
  """Create a set of S2 prototypes with uniformly random entries.
  num_prototypes -- (int) the number of S2 prototype arrays to create
  """
  if __VERBOSE:
    print "Making %d random prototypes" % num_prototypes
  result = GetExperiment().MakeRandomS2Prototypes(num_prototypes)
  if __VERBOSE:
    print "  done: %s s" % GetExperiment().prototype_construction_time
  return result

def SetS2Prototypes(prototypes):
  """Set the S2 prototypes from an array or a file.
  prototypes -- (ndarray) the set of prototypes, or (str) a path to a file
                containing the prototypes
  """
  if isinstance(prototypes, basestring):
    prototypes = util.Load(prototypes)
  elif not isinstance(prototypes, np.ndarray):
    raise ValueError("Please specify an array of prototypes, or the path to a "
        "file.")
  GetExperiment().SetS2Prototypes(prototypes)

def SetCorpus(corpus_dir, classes = None):
  """Read images from the corpus directory, and choose training and testing
  subsets automatically. Use this instead of SetTrainTestSplit().
  corpus_dir -- (str) path to corpus directory
  classes -- (list) set of class names. Use this to ensure a given order to
             the SVM classes. When applying a binary SVM, the first class is
             treated as positive and the second class is treated as negative.
  """
  return GetExperiment().SetCorpus(corpus_dir, classes)

def SetTrainTestSplitFromDirs(train_dir, test_dir, classes = None):
  """Read images from the corpus directories, setting the training and testing
  subsets manually. Use this instead of SetCorpus().
  """
  return GetExperiment().SetTrainTestSplit(train_dir, test_dir, classes)

def SetTrainTestSplit(train_images, test_images, classes):
  """Manually specify the training and testing images."""
  return GetExperiment().SetTrainTestSplit(train_images, test_images, classes)

def ComputeFeatures():
  """Compute SVM feature vectors for all images. Generally, you do not need to
  call this method yourself, as it will be called automatically by RunSvm()."""
  GetExperiment().ComputeFeatures()

def RunSvm(cross_validate = False):
  """Train and test an SVM classifier from the set of images in the corpus.
  cross_validate -- (bool) if true, perform 10x10-way cross-validation.
                    Otherwise, compute accuracy for existing training/testing
                    split.
  RETURN (float tuple) training and testing accuracies
  """
  global __VERBOSE
  e = GetExperiment()
  if cross_validate:
    if __VERBOSE:
      print "Computing cross-validated SVM performance on %d images" % \
          sum(map(len, e.images))
  else:
    if __VERBOSE:
      print "Train SVM on %d images" % sum(map(len, e.train_images))
      print "  and testing on %d images" % sum(map(len, e.test_images))
  train_accuracy, test_accuracy = e.RunSvm(cross_validate)
  if __VERBOSE:
    print "  done: %s s" % e.svm_time
    print "Time to compute feature vectors: %s s" % \
        e.compute_feature_time
  return train_accuracy, test_accuracy

def StoreExperiment(root_path):
  """Save the experiment to disk."""
  return GetExperiment().Store(root_path)

def LoadExperiment(root_path):
  """Load the experiment from disk."""
  global __EXP
  __EXP = Experiment.Load(root_path)
  return __EXP

def Verbose(flag):
  """Set (or unset) verbose logging."""
  global __VERBOSE
  __VERBOSE = flag

#### CLI Interface ####

def CLIGetModel(model_name):
  models = __import__("glimpse.models.%s" % model_name, globals(), locals(),
      ['Model'], 0)
  try:
    return getattr(models, 'Model')
  except AttributeError:
    raise util.UsageException("Unknown model (-m): %s" % model_name)

def CLIMakeClusterPool(config_file = None):
  from glimpse.pools.cluster import ClusterConfig, ClusterPool
  if config_file == None:
    if 'GLIMPSE_CLUSTER_CONFIG' not in os.environ:
      raise util.UsageException("Please specify a cluster configuration file.")
    config_file = os.environ['GLIMPSE_CLUSTER_CONFIG']
  return ClusterPool(ClusterConfig(config_file))

def CLIInit(pool_type = None, cluster_config = None, model_name = None,
    params = None, edit_params = False, layer = None, debug = False,
    verbose = 0, **opts):
  if verbose > 0:
    Verbose(True)
    if verbose > 1:
      logging.getLogger().setLevel(logging.INFO)
  # Make the worker pool
  if pool_type != None:
    pool_type = pool_type.lower()
    if pool_type in ('c', 'cluster'):
      pool = CLIMakeClusterPool(cluster_config)
    elif pool_type in ('m', 'multicore'):
      pool = pools.MulticorePool()
    elif pool_type in ('s', 'singlecore'):
      pool = pools.SinglecorePool()
    else:
      raise util.UsageException("Unknown pool type: %s" % pool_type)
    SetPool(pool)
  if model_name != None:
    SetModelClass(CLIGetModel(model_name))
  model = None
  if edit_params:
    if params == None:
      params = MakeParams()
      params.configure_traits()
      model = MakeModel(params)
  elif params != None:
    model = MakeModel(params)
  if model != None or layer != None:
    SetExperiment(model = model, layer = layer)
  GetExperiment().debug = debug

def CLIRun(prototypes = None, prototype_algorithm = None, num_prototypes = 10,
    corpus = None, svm = False, compute_features = False, result_path = None,
    cross_validate = False, **opts):
  if corpus != None:
    SetCorpus(corpus)
  num_prototypes = int(num_prototypes)
  if prototypes != None:
    SetS2Prototypes(prototypes)
  elif prototype_algorithm != None:
    prototype_algorithm = prototype_algorithm.lower()
    if prototype_algorithm == 'imprint':
      ImprintS2Prototypes(num_prototypes)
    elif prototype_algorithm == 'random':
      MakeRandomS2Prototypes(num_prototypes)
    else:
      raise util.UsageException("Invalid prototype algorithm "
          "(%s), expected 'imprint' or 'random'." % prototype_algorithm)
  if compute_features:
    ComputeFeatures()
  if svm:
    train_accuracy, test_accuracy = RunSvm(cross_validate)
    if not cross_validate:
      print "Train Accuracy: %.3f" % train_accuracy
    print "Test Accuracy: %.3f" % test_accuracy
  if result_path != None:
    StoreExperiment(result_path)

def CLI(**opts):
  """Entry point for command-line interface handling."""
  CLIInit(**opts)
  CLIRun(**opts)

def main():
  default_model = "viz2"
  try:
    opts = dict()
    opts['verbose'] = 0
    result_path = None
    verbose = 0
    cli_opts, cli_args = util.GetOptions('c:C:del:m:n:o:p:P:r:st:vx',
        ['corpus=', 'cluster-config=', 'compute-features', 'debug',
        'edit-options', 'layer=', 'model=', 'num-prototypes=', 'options=',
        'prototype-algorithm=', 'prototypes=', 'results=', 'svm', 'pool-type=',
        'verbose', 'cross-validate'])
    for opt, arg in cli_opts:
      if opt in ('-c', '--corpus'):
        opts['corpus'] = arg
      elif opt in ('-C', '--cluster-config'):
        # Use a cluster of worker nodes
        opts['cluster_config'] = arg
      elif opt in ('--compute-features'):
        opts['compute_features'] = True
      elif opt in ('-d', '--debug'):
        opts['debug'] = True
      elif opt in ('-e', '--edit-options'):
        opts['edit_params'] = True
      elif opt in ('-l', '--layer'):
        opts['layer'] = arg
      elif opt in ('-m', '--model'):
        # Set the model class
        if arg == 'default':
          arg = default_model
        opts['model_name'] = arg
      elif opt in ('-n', '--num-prototypes'):
        opts['num_prototypes'] = int(arg)
      elif opt in ('-o', '--options'):
        opts['params'] = util.Load(arg)
      elif opt in ('-p', '--prototype-algorithm'):
        opts['prototype_algorithm'] = arg.lower()
      elif opt in ('-P', '--prototypes'):
        opts['prototypes'] = util.Load(arg)
      elif opt in ('-r', '--results'):
        opts['result_path'] = arg
      elif opt in ('-s', '--svm'):
        opts['svm'] = True
      elif opt in ('-t', '--pool-type'):
        opts['pool_type'] = arg.lower()
      elif opt in ('-v', '--verbose'):
        opts['verbose'] += 1
      elif opt in ('-x', '--cross-validate'):
        opts['cross_validate'] = True
    CLI(**opts)
  except util.UsageException, e:
    util.Usage("[options]\n"
        "  -c, --corpus=DIR                Use corpus directory DIR\n"
        "  -C, --cluster-config=FILE       Read cluster configuration from "
        "FILE\n"
        "      --compute-features          Compute feature vectors (implied "
        "by -s)\n"
        "  -d, --debug                     Enable debugging\n"
        "  -e, --edit-options              Edit model options with a GUI\n"
        "  -l, --layer=LAYR                Compute feature vectors from LAYR "
        "activity\n"
        "  -m, --model=MODL                Use model named MODL\n"
        "  -n, --num-prototypes=NUM        Generate NUM S2 prototypes\n"
        "  -o, --options=FILE              Read model options from FILE\n"
        "  -p, --prototype-algorithm=ALG   Generate S2 prototypes according "
        "to algorithm\n"
        "                                  ALG (one of 'imprint' or 'random')\n"
        "  -P, --prototypes=FILE           Read S2 prototypes from FILE "
        "(overrides -p)\n"
        "  -r, --results=FILE              Store results to FILE\n"
        "  -s, --svm                       Train and test an SVM classifier\n"
        "  -t, --pool-type=TYPE            Set the worker pool type\n"
        "  -v, --verbose                   Enable verbose logging\n"
        "  -x, --cross-validate            Compute test accuracy via cross-"
        "validation\n"
        "                                  instead of fixed training/testing "
        "split",
        e
    )

if __name__ == '__main__':
  main()
