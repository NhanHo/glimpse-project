"""This module implements a two-stage HMAX-like model.

This module implements a multi-scale analysis by applying single-scale Gabors to
a scale pyramid of the input image. This is similar to the configuration used by
Mutch & Lowe (2008).

"""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import numpy as np
from scipy.ndimage.interpolation import zoom

from glimpse.backends import BackendException, InsufficientSizeException
from glimpse.models.misc import BaseState, Whiten
from glimpse.models.viz2.model import Model as Viz2Model
from glimpse.models.viz2.model import Layer
from glimpse.util import kernel
from glimpse.util import gimage
from .params import Params

class State(BaseState):
  """A container for the :class:`Model` state."""
  pass

def MinimumRetinaSize(p):
  """Compute the smallest retinal layer that supports the given parameters.

  This function discounts the effect of scaling.

  :param p: Parameter settings for model.
  :rtype: int
  :returns: Length of smaller edge of retina.

  """
  if p.operation_type == "valid":
    # Must support at least one S2 unit
    c1_size = p.s2_kwidth
    s1_size = c1_size * p.c1_sampling + p.c1_kwidth - 1
    retina_size = s1_size * p.s1_sampling + p.s1_kwidth - 1
  else:  # centered convolution
    c1_size = (p.s2_kwidth - 1) / 2
    s1_size = c1_size * p.c1_sampling + (p.c1_kwidth - 1) / 2
    retina_size = s1_size * p.s1_sampling + (p.s1_kwidth - 1) / 2
  return retina_size

def NumScalesSupported(params, retina_size):
  """Compute the number of scale bands supported for a given retinal size.

  This ensures that at least one S2 unit can be computed for every scale band.

  :param params: Parameter settings for model.
  :param retina_size: Length of shorter edge of retina layer.
  :type retina_size: int
  :rtype: int
  :return: Number of scales.

  """
  min_retina_size = MinimumRetinaSize(params)
  num_scales = 0
  while min_retina_size < retina_size:
    num_scales += 1
    min_retina_size *= params.scale_factor
  return num_scales

class Model(Viz2Model):
  """Create a 2-part, HMAX-like hierarchy of S+C layers."""

  #: The datatype associated with layer descriptors for this model.
  LayerClass = Layer

  #: The parameters type associated with this model.
  ParamClass = Params

  #: The datatype associated with network states for this model.
  StateClass = State

  @property
  def s1_kernel_shape(self):
    """The expected shape of the S1 kernels array, including band structure.

    :rtype: tuple of int

    """
    p = self.params
    return p.s1_num_orientations, p.s1_num_phases, p.s1_kwidth, p.s1_kwidth

  @property
  def s1_kernels(self):
    """The set of S1 kernels, which is generated if not set.

    :returns: S1 kernels indexed by orientation, and phase.
    :rtype: 4D ndarray of float

    """
    # if kernels array is empty, then generate it using current model parameters
    if self._s1_kernels == None:
      p = self.params
      self._s1_kernels = kernel.MakeGaborKernels(
          kwidth = p.s1_kwidth,
          num_orientations = p.s1_num_orientations,
          num_phases = p.s1_num_phases, shift_orientations = True,
          scale_norm = self.s1_kernels_are_normed)
    return self._s1_kernels

  def BuildS1FromRetina(self, retina):
    """Apply S1 processing to some existing retinal layer data.

    .. note::

       This method pools over phase, so the output has only scale and
       orientation bands.

    :param retina: Result of retinal layer processing.
    :type retina: 2D ndarray of float
    :return: S1 maps indexed by scale and orientation.
    :rtype: list of 3D ndarray of float

    """
    # Create scale pyramid of retinal map
    p = self.params
    num_scales = p.num_scales
    if num_scales == 0:
      num_scales = NumScalesSupported(p, min(retina.shape))
    retina_scales = gimage.MakeScalePyramid(retina, num_scales,
        1.0 / p.scale_factor)
    ndp = (p.s1_operation == 'NormDotProduct')
    s1_kernels = self.s1_kernels
    if ndp:
      # NDP is already phase invariant, just use one phase of filters
      s1_kernels = self.s1_kernels[:, 0].copy()
    # Reshape kernel array to be 3-D: index, 1, y, x
    s1_kernels = s1_kernels.reshape((-1, 1, p.s1_kwidth, p.s1_kwidth))
    s1s = []
    backend_op = getattr(self.backend, p.s1_operation)
    for scale in range(num_scales):
      # Reshape retina to be 3D array
      retina = retina_scales[scale]
      retina_ = retina.reshape((1,) + retina.shape)
      try:
        s1 = backend_op(retina_, s1_kernels, bias = p.s1_bias, beta = p.s1_beta,
            scaling = p.s1_sampling)
      except InsufficientSizeException, ex:
        ex.message = "Image is too small to apply S1 filters at scale %d" % scale
	ex.scale = scale
        raise
      except BackendException, ex:
        ex.scale = scale
        raise
      if ndp:
        np.abs(s1, s1)  # Take the absolute value in-place
        # S1 is now a 3D array of phase-invariant responses
      else:
        # Reshape S1 to be 4D array
        s1 = s1.reshape((p.s1_num_orientations, p.s1_num_phases) + \
            s1.shape[-2:])
        # Pool over phase.
        s1 = s1.max(1)
      if np.isnan(s1).any():
        raise BackendException("Found illegal values in S1 map at scale %d" % \
            scale)
      # Append 3D array to list
      s1s.append(s1)
    return s1s

  def BuildC1FromS1(self, s1s):
    """Compute the C1 layer activity from multi-scale S1 activity.

    :param s1s: S1 maps indexed by scale.
    :type s1s: list of 3D ndarray of float, or 4D ndarray of float
    :returns: C1 maps indexed by scale and orientation.
    :rtype: list of 3D ndarray of float

    """
    p = self.params
    c1s = [ self.backend.LocalMax(s1, kwidth = p.c1_kwidth,
        scaling = p.c1_sampling) for s1 in s1s ]
    if p.c1_whiten:
      # Whiten each scale independently, modifying values in-place.
      map(Whiten, c1s)
    return c1s

  def BuildS2FromC1(self, c1s):
    """Compute the S2 layer activity from multi-scale C1 activity.

    :param c1s: C1 maps indexed by scale and orientation.
    :type c1s: 4D ndarray of float, or list of 3D ndarray of float
    :returns: S2 maps indexed by scale and prototype.
    :rtype: list of 3D ndarray of float

    """
    if self.s2_kernels == None or len(self.s2_kernels[0]) == 0:
      raise Exception("Need S2 kernels to compute S2 layer activity, but none "
          "were specified.")
    kernels = self.s2_kernels[0]
    if len(c1s) == 0:
      return []
    p = self.params
    s2s = []
    backend_op = getattr(self.backend, p.s2_operation)
    for scale in range(len(c1s)):
      c1 = c1s[scale]
      try:
        s2 = backend_op(c1, kernels, bias = p.s2_bias, beta = p.s2_beta,
            scaling = p.s2_sampling)
      except BackendException, ex:
        # Annotate exception with scale information.
        ex.scale = scale
        raise
      if np.isnan(s2).any():
        raise BackendException("Found illegal values in S2 map at scale %d" % \
            scale)
      # Append 3D array to list.
      s2s.append(s2)
    return s2s

# Add (circular) Model reference to State class.
State.ModelClass = Model
