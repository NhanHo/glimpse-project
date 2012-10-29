import glimpse.util.pil_fix # must come before sklearn imports

import logging
import numpy as np
import operator
import os
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sys
import time
import types
import copy

from glimpse.models import misc
from glimpse import util
from glimpse.util.grandom import HistogramSampler
from glimpse.util import svm
from glimpse.models.misc import InputSourceLoadException
from glimpse.backends import BackendException, InsufficientSizeException
from traits.api import  Str, Int, List, Button, HasTraits
from traitsui.api import View, Item, CheckListEditor, Group, Handler, Action, TextEditor
import misc as pmisc

class Experiment(object):

  def __init__(self,model,corpus,classifier,pool):
    assert model != None
    assert layer != None
    assert pool != None
    
    Corpus = ExpCorpus()
    Model = ExpModel()
    Classifier = ExpClassifier()
    #: The worker pool used for parallelization.
    self.pool = pool

  def Store(self, root_path):
    """Save the experiment to disk."""
    pool = self.pool
    self.pool = None  # can't serialize some pools
    util.Store(self, root_path)
    self.pool = pool

  @staticmethod
  def Load(root_path):
    """Load the experiment from disk.

    :returns: The experiment object.
    :rtype: Experiment

    """
    experiment = util.Load(root_path)
    return experiment

class ExpCorpus(HasTraits):
  def __repr__(self):
    return self.name

  def __init__(self, corpus_dir = None, images = None, classes = None, name = None):
    print "Corpus object initialization"
    #: (str): path to corpus directory
    self.corpus_dir = corpus_dir
    #: (list of list of ...) The set of data of this corpus, the data could be
    #raw image, or a feature layer
    self.dir_reader = pmisc.DirReader(ignore_hidden = True)    
    self.images = images
    self.classes = classes 
    self.name = name
   
    self.features = None
    if images == None:
      print "flag"
      print os.getcwd()
      print corpus_dir
      self.SetCorpus(corpus_dir)
    #: (str): Name of classes
    #: (str) Path to the corpus directory. May be empty if
    #: :meth:`SetCorpusSubdirs` was used.
    #: The file-system reader.
    self.view = View (Item('name'),title = 'Model')
  def GetFeatures(self):
    """The full set of features for each class, without training/testing splits.

    :rtype: list of 2D ndarray of float
    :returns: Copy of feature data, indexed by class, image, and then feature
       offset.

    """
    if self.train_features == None:
      return None
    # Reorder instances from (set, class) indexing, to (class, set) indexing.
    features = zip(self.train_features, self.test_features)
    # Concatenate instances for each class (across sets)
    features = map(np.vstack, features)
    return features

  def GetImages(self):
    """The full set of images, without training/testing splits.

    :rtype: list of list of str
    :returns: Copy of image paths, indexed by class, and then image.

    """
    if self.train_images == None:
      return None
    # Combine images by class, and concatenate lists.
    return map(util.UngroupLists, zip(self.train_images, self.test_images))
    

  def SetCorpus(self, corpus_dir, classes = None, balance = False):
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

  def SetCorpusSubdirs(self, corpus_subdirs, corpus = None, classes = None,
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
    self.classes = classes
    self.corpus = corpus
    self.train_test_split = 'automatic'
    try:
      images_per_class = map(self.dir_reader.ReadFiles, corpus_subdirs)
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
    self.train_images = [ images[ : len(images)/2 ]
        for images in images_per_class ]
    self.test_images = [ images[ len(images)/2 : ]
        for images in images_per_class ]

  def SetTrainTestSplitFromDirs(self, train_dir, test_dir, classes = None):
    """Read images from the corpus directories.

    Training and testing subsets are chosen manually.

    :param train_dir: Path to directory of training images.
    :type train_dir: str
    :param test_dir: Path to directory of testing images.
    :type test_dir: str
    :param classes: Class names.
    :type classes: list of str

    .. seealso::
       :func:`SetCorpus`

    """
    try:
      if classes == None:
        classes = map(os.path.basename, self.dir_reader.ReadDirs(train_dir))
      train_images = map(self.dir_reader.ReadFiles,
          [ os.path.join(train_dir, cls) for cls in classes ])
      test_images = map(self.dir_reader.ReadFiles,
          [ os.path.join(test_dir, cls) for cls in classes ])
    except OSError, ex:
      sys.exit("Failed to read corpus directory: %s" % ex)
    self.SetTrainTestSplit(train_images, test_images, classes)
    self.corpus = (train_dir, test_dir)

  def SetTrainTestSplit(self, train_images, test_images, classes):
    """Manually specify the training and testing images.

    :param train_images: Paths for each training image, with one sub-list per
       class.
    :type train_images: list of list of str
    :param test_images: Paths for each training image, with one sub-list per
       class.
    :type test_images: list of list of str
    :param classes: Class names
    :type classes: list of str

    """
    if classes == None:
      raise ValueError("Must specify set of classes.")
    if train_images == None:
      raise ValueError("Must specify set of training images.")
    if test_images == None:
      raise ValueError("Must specify set of testing images.")
    self.classes = classes
    self.train_test_split = 'manual'
    self.train_images = train_images
    self.test_images = test_images

  def SetFeatures(self, train_features, test_features):
    self.train_features = train_features
    self.test_features = test_features

  def GetFeatures(self):
    return self.features

class ExpModel(HasTraits):
  
  #To avoid layer_str_changed being triggered every keystroke
  layer_str = Str("C2",editor = TextEditor(auto_set = False))
  EditParamsButton = Button ('Edit Params')
  traits_view = View (Item('name'),
                      Item('layer_str'), 
                      buttons = ['OK'], kind = 'modal')
  def _layer_str_changed(self):
    temp = str(self.layer_str)
    layer = pmisc.MakeDefaultLayer(temp)
    self.layer = layer
    
  def __str__ (self):
    return self.name

  def __repr__ (self):
    return self.name
    
  def __init__(self, model = None, layer = None, pool = None, params = None, prototype_source = None, name = "Model"):
    DefaultModel = pmisc.MakeModel(model, layer, pool, params, prototype_source)
    model = DefaultModel[0]
    layer = DefaultModel[1]
    pool = DefaultModel[2]
    self.name = name
    self.model = model
    #: The model layer from which feature vectors are extracted.
    self.layer = layer 
    #The worker pool which the model will use
    self.pool = pool
    #: (str) The method used to construct prototypes.
    self.prototype_source = None
    #: (float) Time required to build S2 prototypes, in seconds.
    self.prototype_construction_time = None
    #: (2D list of 4-tuple) Patch locations for imprinted prototypes.
    self.prototype_imprint_locations = None
    #: (float) Time elapsed while constructing feature vectors (in seconds).
    self.compute_feature_time = None

  @property
  def s2_prototypes(self):
    """The set of S2 prototypes."""
    return self.model.s2_kernels

  @s2_prototypes.setter
  def s2_prototypes(self, value):
    self.prototype_source = 'manual'
    self.model.s2_kernels = value

  def ImprintS2Prototypes(self, corpus, num_prototypes):
    """Imprint a set of S2 prototypes from a corpus.

    Patches are drawn from all classes of the training data.

    :param int num_prototypes: The number of C1 patches to sample.

    """
    model = self.model
    start_time = time.time()
    image_files = util.UngroupLists(corpus.images)
    # Represent each image file as an empty model state.
    input_states = map(model.MakeStateFromFilename, image_files)
    try:
      prototypes, locations = misc.ImprintKernels(model, model.LayerClass.C1,
          model.s2_kernel_sizes, num_prototypes, input_states,
          normalize = model.s2_kernels_are_normed, pool = self.pool)
    except InputSourceLoadException, ex:
      if ex.source != None:
        path = ex.source.image_path
      else:
        path = '<unknown>'
      logging.error("Failed to process image (%s): image read error", path)
      sys.exit(-1)
    except InsufficientSizeException, ex:
      if ex.source != None:
        path = ex.source.image_path
      else:
        path = '<unknown>'
      logging.error("Failed to process image (%s): image too small", path)
      sys.exit(-1)
    # Store new prototypes in model.
    self.prototype_source = 'imprinted'
    # Convert input source index to corresponding image path.
    locations = [ [ (image_files[loc[0]],) + loc[1:]
        for loc in locs_for_ksize ] for locs_for_ksize in locations ]
    self.prototype_imprint_locations = locations
    model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time

  def MakeUniformRandomS2Prototypes(self, corpus, num_prototypes, low = None,
      high = None):
    """Create a set of S2 prototypes with uniformly random entries.

    :param int num_prototypes: The number of S2 prototype arrays to create.

    The function does not use any corpus for Prototypes, the parameter is put there for consistency 
    with other functions' interface
    """
    if low == None:
      low = 0
    if high == None:
      high = 1
    start_time = time.time()
    model = self.model
    prototypes = []
    for kshape in model.s2_kernel_shapes:
      shape = (num_prototypes,) + tuple(kshape)
      prototypes_for_size = np.random.uniform(low, high, shape)
      if model.s2_kernels_are_normed:
        for proto in prototypes_for_size:
          proto /= np.linalg.norm(proto)
      prototypes.append(prototypes_for_size)
    model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'uniform'

  def MakeShuffledRandomS2Prototypes(self, corpus, num_prototypes):
    """Create a set of S2 prototypes.

    The set is chosen by imprinting, and then shuffling the order of entries
    within each prototype.

    :param int num_prototypes: The number of S2 prototype arrays to create.

    """
    start_time = time.time()
    if self.model.s2_kernels == None:
      self.ImprintS2Prototypes(num_prototypes)
    for kernels_for_size in self.model.s2_kernels:
      for kernel in kernels_for_size:
        np.random.shuffle(kernel.flat)
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'shuffle'

  def MakeHistogramRandomS2Prototypes(self, corpus, num_prototypes):
    """Create a set of S2 prototypes.

    The set is created by drawing elements from a distribution, which is
    estimated from a set of imprinted prototypes. Each entry is drawn
    independently of the others.

    :param int num_prototypes: The number of S2 prototype arrays to create.

    """
    start_time = time.time()
    model = self.model
    # We treat each kernel size independently. We want the histogram for each
    # kernel size to be based on ~100k C1 samples minimum. Here, we calculate
    # the number of imprinted prototypes required to get this.
    c1_samples_per_prototype = min([ reduce(operator.mul, shape)
        for shape in model.s2_kernel_shapes ])
    num_desired_c1_samples = 100000
    num_imprinted_prototypes = int(num_desired_c1_samples /
        float(c1_samples_per_prototype))
    self.ImprintS2Prototypes(corpus, num_prototypes = num_imprinted_prototypes)
    # For each kernel size, build a histogram and sample new prototypes from it.
    prototypes = []
    for idx in range(len(model.s2_kernel_shapes)):
      kernels = model.s2_kernels[idx]
      shape = model.s2_kernel_shapes[idx]
      hist = HistogramSampler(kernels.flat)
      size = (num_prototypes,) + shape
      prototypes_for_size = hist.Sample(size, dtype = util.ACTIVATION_DTYPE)
      if model.s2_kernels_are_normed:
        for proto in prototypes_for_size:
          proto /= np.linalg.norm(proto)
      prototypes.append(prototypes_for_size)
    model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'histogram'

  def MakeNormalRandomS2Prototypes(self, corpus, num_prototypes):
    """Create a set of S2 prototypes.

    The set is created by drawing elements from the normal distribution, whose
    parameters are estimated from a set of imprinted prototypes. Each entry is
    drawn independently of the others.

    :param int num_prototypes: The number of S2 prototype arrays to create.

    """
    start_time = time.time()
    model = self.model
    # We treat each kernel size independently. We want the histogram for each
    # kernel size to be based on ~100k C1 samples minimum. Here, we calculate
    # the number of imprinted prototypes required to get this.
    c1_samples_per_prototype = min([ reduce(operator.mul, shape)
        for shape in model.s2_kernel_shapes ])
    num_desired_c1_samples = 100000
    num_imprinted_prototypes = int(num_desired_c1_samples /
        float(c1_samples_per_prototype))
    self.ImprintS2Prototypes(corpus, num_prototypes = num_imprinted_prototypes)
    # For each kernel size, estimate parameters of a normal distribution and
    # sample from it.
    prototypes = []
    for idx in range(len(model.s2_kernel_shapes)):
      kernels = model.s2_kernels[idx]
      shape = model.s2_kernel_shapes[idx]
      mean, std = kernels.mean(), kernels.std()
      size = (num_prototypes,) + shape
      prototypes_for_size = np.random.normal(mean, std, size = size).astype(
          util.ACTIVATION_DTYPE)
      if model.s2_kernels_are_normed:
        for proto in prototypes_for_size:
          proto /= np.linalg.norm(proto)
      prototypes.append(prototypes_for_size)
    model.s2_kernels = prototypes
    self.prototype_construction_time = time.time() - start_time
    self.prototype_source = 'normal'

  def SetS2Prototypes(self, prototypes):
    """Set the S2 prototypes from an array.

    :param ndarray prototypes: The set of prototypes.

    """
    self.prototype_source = 'manual'
    self.model.s2_kernels = prototypes

  def GetImageFeatures(self, images, raw = False, save_all = False,
      block = True):
    """Return the activity of the model's output layer for a set of images.

    :param input_states: Image paths.
    :type input_states: str, or iterable of str
    :param bool raw: Whether to return per-image results as a single feature
       vector or the raw state object.
    :param bool save_all: Whether resulting states should contain values for
       all computed layers in the network. See :meth:`BuildLayer
       <glimpse.models.misc.BaseModel.BuildLayer>`.
    :param bool block: (experimental) Block while waiting for results.
    :rtype: iterable of 1D ndarray of float
    :returns: A feature vector for each image.

    """
    states = map(self.model.MakeStateFromFilename, images)
    return self.GetStateFeatures(states, raw = raw, save_all = save_all,
        block = block)

  def GetStateFeatures(self, input_states, raw = False, save_all = False,
      block = True):
    """Return the activity of the model's output layer for a set of states.

    :param input_states: Model states containing image data.
    :type input_states: iterable of State
    :param bool raw: Whether to return per-image results as a single feature
       vector or the raw state object.
    :param bool save_all: Whether resulting states should contain values for
       all computed layers in the network. See :meth:`BuildLayer
       <glimpse.models.misc.BaseModel.BuildLayer>`.
    :param bool block: (experimental) Block while waiting for results.
    :returns: Features for each image.
    :rtype: iterable of 1D ndarray of float, or iterable of state (if raw is
       True)

    """
    if self.model.s2_kernels == None:
      lyr = self.model.LayerClass
      if lyr.IsSublayer(lyr.S2, self.layer):
        sys.exit("Please set the S2 prototypes before computing feature vectors"
            " for layer %s." % self.layer.name)
    builder = self.model.BuildLayerCallback(self.layer, save_all = save_all)
    # Compute model states containing desired features.
    output_states = self.pool.imap(builder, input_states)
    # Look up the activity values for the output layer, and convert them all to
    # a single vector.
    layer = self.layer.ident
    if raw:
      features = output_states
    else:
      features = ( util.FlattenArrays(st[layer]) for st in output_states )
    if block:
      try:
        features = list(features)  # wait for results
      except InputSourceLoadException, ex:
        logging.error("Failed to read image from disk: %s",
            ex.source.image_path)
        sys.exit(-1)
      except InsufficientSizeException, ex:
        logging.error("Failed to process image (%s): image too small (%s)",
            ex.source.image_path, ex.message)
        sys.exit(-1)
      except BackendException, ex:
        if ex.source == None:
          path = "unknown image"
        else:
          path = ex.source.image_path
        logging.error("Failed to process image (%s): %s", path, ex.message)
        sys.exit(-1)
    return features
  
  def ComputeFeatures(self):
    """Compute SVM feature vectors for all images.

    Generally, you do not need to call this method yourself, as it will be
    called automatically by :meth:`RunSvm`.

    """
    if self.train_images == None or self.test_images == None:
      sys.exit("Please specify the corpus.")
    train_sizes = map(len, self.train_images)
    train_size = sum(train_sizes)
    test_sizes = map(len, self.test_images)
    test_size = sum(test_sizes)
    train_images = util.UngroupLists(self.train_images)
    test_images = util.UngroupLists(self.test_images)
    images = train_images + test_images
    start_time = time.time()
    # Compute features for all images.
    features = self.GetImageFeatures(images, block = True)
    self.compute_feature_time = time.time() - start_time
    # Split results by training/testing set
    train_features, test_features = util.SplitList(features,
        [train_size, test_size])
    # Split training set by class
    train_features = util.SplitList(train_features, train_sizes)
    # Split testing set by class
    test_features = util.SplitList(test_features, test_sizes)
    # Store features as list of 2D arrays
    self.train_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in train_features ]
    self.test_features = [ np.array(f, util.ACTIVATION_DTYPE)
        for f in test_features ]

  
  def ComputeFeatures(self, corpus):
    """Compute SVM feature vectors for all images.

    Generally, you do not need to call this method yourself, as it will be
    called automatically by :meth:`RunSvm`.

    """
    if corpus.images == None:
      sys.exit("Please specify the corpus.")
    image_sizes = map(len, corpus.images)
    image_size = sum(image_sizes)
    images = util.UngroupLists(corpus.images)
    start_time = time.time()
    features = self.GetImageFeatures(images, block = True)
    self.compute_feature_time = time.time() - start_time
    features = util.SplitList(features, image_sizes)
    features = [np.array(f,util.ACTIVATION_DTYPE) for f in features]
    corpus.features = features
    return features
    
class ExpClassifier(HasTraits):
  def __repr__(self):
    return self.name

  def __init__(self, name = "Classifier"):
    self.name = name
    #: (dict) SVM evaluation results on the training data.
    self.train_results = None
    #: (dict) SVM evaluation results on the test data.
    self.test_results = None
    #: (bool) Flag indicating whether cross-validation was used to compute test
    #: accuracy.
    self.cross_validated = False
    #: (float) Time required to train the SVM, in seconds.
    self.svm_train_time = None
    #: (float) Time required to test the SVM, in seconds.
    self.svm_test_time = None
    #: (float) Time elapsed while training and testing the SVM (in seconds).
    self.svm_time = None
    #: The built SVM classifier.
    self.classifier = None

  def TrainSvm(self, corpus):
    """Construct an SVM classifier from the set of training images.

    :returns: Training accuracy.
    :rtype: float

    """
    self.train_features = corpus.GetFeatures()
    # Prepare the data
    train_features, train_labels = svm.PrepareFeatures(self.train_features)
    # Create the SVM classifier with feature scaling.
    self.classifier = svm.Pipeline([ ('scaler', sklearn.preprocessing.Scaler()),
        ('svm', sklearn.svm.LinearSVC())])
    self.classifier.fit(train_features, train_labels)
    # Evaluate the classifier
    decision_values = self.classifier.decision_function(train_features)
    predicted_labels = self.classifier.predict(train_features)
    accuracy = sklearn.metrics.zero_one_score(train_labels, predicted_labels)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(train_labels, predicted_labels)
    auc = sklearn.metrics.auc(fpr, tpr)
    self.train_results = dict(decision_values = decision_values,
        predicted_labels = predicted_labels, accuracy = accuracy, auc = auc)
    return self.train_results['accuracy']

  def TestSvm(self, corpus):
    """Test a learned SVM classifier.

    The classifier is applied to the set of test images.

    :returns: Test accuracy.
    :rtype: float

    """
    # Prepare the data
    test_features, test_labels = svm.PrepareFeatures(corpus.GetFeatures())
    # Evaluate the classifier
    decision_values = self.classifier.decision_function(test_features)
    predicted_labels = self.classifier.predict(test_features)
    accuracy = sklearn.metrics.zero_one_score(test_labels, predicted_labels)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_labels, predicted_labels)
    auc = sklearn.metrics.auc(fpr, tpr)
    self.test_results = dict(decision_values = decision_values,
        predicted_labels = predicted_labels, accuracy = accuracy, auc = auc)
    return self.test_results['accuracy']

  def CrossValidateSvm(self):
    """Test a learned SVM classifier.

    The classifier is applied to all images using 10-by-10-way cross-validation.

    :returns: Cross-validation accuracy
    :rtype: float

    """
    if self.train_features == None:
      self.ComputeFeatures()
    features, labels = svm.PrepareFeatures(self.GetFeatures())
    # Create the SVM classifier with feature scaling.
    classifier = svm.Pipeline([ ('scaler', sklearn.preprocessing.Scaler()),
        ('svm', sklearn.svm.LinearSVC())])
    scores = sklearn.cross_validation.cross_val_score(classifier, features,
        labels, cv = 10, n_jobs = -1)
    test_accuracy = scores.mean()
    self.train_results = None
    self.test_results = dict(accuracy = test_accuracy)
    self.cross_validated = True
    return test_accuracy

  def RunSvm(self, cross_validate = False):
    """Train and test an SVM classifier.

    The classifier is trained using the set of training images. Accuracy for
    this classifier is reported for each of the training and testing sets.

    :param bool cross_validate: If true, perform 10x10-way cross-validation.
       Otherwise, compute accuracy for existing training/testing split.
    :returns: Training and testing accuracies. (Training accuracy is None when
       using cross-validation).
    :rtype: 2-tuple of float

    """
    if self.train_features == None:
      self.ComputeFeatures()
    start_time = time.time()
    if cross_validate:
      self.CrossValidateSvm()
      train_accuracy = None
    else:
      self.TrainSvm()
      self.TestSvm()
      train_accuracy = self.train_results['accuracy']
    self.svm_time = time.time() - start_time
    return train_accuracy, self.test_results['accuracy']
    
