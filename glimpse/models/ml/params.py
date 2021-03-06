# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

from glimpse.models.misc import BaseParams
from glimpse.models.viz2.params import KWidth, SLayerOperation
from glimpse.util import traits

class OperationType(traits.Enum):
  """A trait type describing how S- and C-layer operations are applied."""

  def __init__(self, value, **metadata):
    super(OperationType, self).__init__(value, ("valid"),  #, "centered"),
        **metadata)

class Params(BaseParams):
  """Parameter container for the :class:`ml model
  <glimpse.models.ml.model.Model>`.

  """

  retina_bias = traits.Range(low = 0., value = 1., label = "Retina Bias",
      desc = "term added to standard deviation of local window")
  retina_enabled = traits.Bool(False, label = "Retina Enabled",
      desc = "indicates whether the retinal layer is used")
  retina_kwidth = KWidth(15, label = "Retina Kernel Width",
      desc = "spatial width of input neighborhood for retinal units")

  s1_beta = traits.Range(low = 0., value = 1., exclude_low = True,
      label = "S1 Beta", desc = "term added to the norm of the input vector")
  s1_bias = traits.Range(low = 0., value = 0.01, label = "S1 Bias",
      desc = "beta parameter of RBF for S1 cells")
  s1_kwidth = KWidth(11, label = "S1 Kernel Width",
      desc = "spatial width of input neighborhood for S1 units")
  s1_num_orientations = traits.Range(low = 1, value = 8,
      label = "Number of Orientations",
      desc = "number of different S1 Gabor orientations")
  s1_num_phases = traits.Range(low = 1, value = 2, label = "Number of Phases",
      desc = "number of different phases for S1 Gabors. Using two phases "
          "corresponds to find a light bar on a dark background and vice versa")
  s1_sampling = traits.Range(low = 1, value = 1, label = "S1 Sampling",
      desc = "subsampling factor (e.g., setting this parameter to 2 will "
      "result in an S1 array that is half the width -- and half the height "
      "-- of the input array)")
  s1_shift_orientations = traits.Bool(True, label = "Shift Orientations",
      desc = "rotate Gabors by a small positive angle")
  s1_operation = SLayerOperation("NormDotProduct", label = "S1 Operation")

  c1_kwidth = KWidth(5, label = "C1 Kernel Width",
      desc = "spatial width of input neighborhood for C1 units")
  c1_sampling = traits.Range(low = 1, value = 2, label = "C1 Sampling",
      desc = "subsampling factor (e.g., setting this parameter to 2 will "
      "result in a C1 array that is half the width -- and half the height "
      "-- of the S1 array)")
  c1_whiten = traits.Bool(False, label = "C1 Whiten",
      desc = "whether to normalize the total energy at each C1 location")

  s2_beta = traits.Range(low = 0., value = 5., exclude_low = True,
      label = "S2 Beta", desc = "beta parameter of RBF for S1 cells")
  # Default value is configured to match distribution of C1 norm under
  # whitening.
  s2_bias = traits.Range(low = 0., value = 0.1, label = "S2 Bias",
      desc = "additive term combined with input window norm")
  s2_kwidth = KWidth(7, label = "S2 Kernel Width",
      desc = "spatial width of input neighborhood for S2 units")
  s2_sampling = traits.Range(low = 1, value = 1, label = "S2 Sampling",
      desc = "subsampling factor (e.g., setting this parameter to 2 will "
      "result in an S2 array that is half the width -- and half the height "
      "-- of the C1 array)")
  s2_operation = SLayerOperation("Rbf", label = "S2 Operation")

  operation_type = OperationType("valid", label = "Operation Type",
      desc = "the way in which S- and C-layers are applied")
  num_scales = traits.Range(low = 0, value = 0, label = "Number of Scales",
      desc = "number of different scale bands (set to zero to use as many as "
          "possible for a given image size)")
  scale_factor = traits.Range(low = 1., value = 2**(1/2.),
      label = "Scaling Factor",
      desc = "Image downsampling factor between scale bands (must be greater "
          "than one)")
