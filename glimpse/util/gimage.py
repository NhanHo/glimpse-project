"""Miscellaneous functions related to images and image processing."""

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

import Image
import numpy as np
from scipy import fftpack
from scipy.misc import fromimage, toimage
import sys

from garray import ACTIVATION_DTYPE, PadArray

def ScaleImage(img, size):
  """Resize an image.

  :param Image img: Input image.
  :param size: Size of output image in the format (width, height).
  :type size: 1D array-like of float or int
  :return: Resized version of input image.
  :rtype: Image

  """
  size = np.array(size, dtype = int)
  # Use bicubic interpolation if the new width is larger than the old width.
  if size[0] > img.size[0]:
    method = Image.BICUBIC  # interpolate
  else:
    method = Image.ANTIALIAS  # blur and down-sample
  return img.resize(size, method)

def ScaleAndCropImage(img, size):
  """Resize an image by scaling and cropping.

  Input is scaled to fit the target dimensions (preserving aspect ratio), and
  then border pixels are removed.

  :param Image img: Input image.
  :param size: Size of output image in the format (width, height).
  :type size: 1D array-like of float or int
  :return: Resized version of input image.
  :rtype: Image

  """
  size = np.array(size, dtype = int)
  img_width, img_height = img.size
  image_rho = img_width / float(img_height)  # aspect ratio of input
  target_width, target_height = size
  target_rho = target_width / float(target_height)  # aspect ratio of output
  if image_rho > target_rho:
    # Scale to target height (maintaining aspect ratio) and crop border pixels
    # from left and right edges. Note that the scaled width is guaranteed to
    # be at least as large as the target width.
    scaled_width = int(float(target_height) * image_rho)
    img = ScaleImage(img, size = (scaled_width, target_height))
    border = int((scaled_width - target_width) / 2.)
    # Bounding box format is left, upper, right, and lower; where the point
    # (0,0) corresponds to the top-left corner of the image.
    img = img.crop(box = (border, 0, border + target_width, target_height))
  else:
    # Scale to target width (maintaining aspect ratio) and crop border pixels
    # from top and bottom edges. Note that the scaled height is guaranteed to
    # be at least as large as the target height.
    scaled_height = int(float(target_width) / image_rho)
    img = ScaleImage(img, size = (target_width, scaled_height))
    border = int((scaled_height - target_height) / 2.)
    # Bounding box format is left, upper, right, and lower; where the point
    # (0,0) corresponds to the top-left corner of the image.
    img = img.crop(box = (0, border, target_width, border + target_height))
  assert np.all(img.size == size), "Result image size is %s, but requested %s" % (img.size, size)
  return img

def ImageToArray(img, array = None, transpose = True):
  """Load image data into a 2D numpy array.

  :param img: Image to read.
  :type img: PIL.Image
  :param ndarray array: Output array. If unspecified, one will be generated
     automatically.
  :param bool transpose: Whether image data should be transposed before
     returning.
  :returns: Array containing image data. Note that this may be non-contiguous.
  :rtype: ndarray

  .. seealso::
     :func:`scipy.misc.misc.fromimage`.

  """
  def MakeBuffer():
    if img.mode == 'L':
      return np.empty(img.size, dtype = np.uint8)
    elif img.mode == 'RGB':
      return np.empty(img.size + (3,), dtype = np.uint8)
    elif img.mode == 'F':
      return np.empty(img.size, dtype = np.float)
    elif img.mode == '1':
      return np.empty(img.size, dtype = np.bool)
    raise Exception("Can't load data from image with mode: %s" % img.mode)
  def CopyImage(dest):
    img_data = img.load()
    for idx in np.ndindex(img.size):
      dest[idx] = img_data[idx]
    return dest
  def CheckArrayShape():
    shape = list(img.size)
    if transpose:
      shape = shape[::-1]
    shape = tuple(shape)
    assert shape == array.shape, "Array has wrong shape: expected %s but got" \
        "%s" % (shape, array.shape)
  if array != None:
    if not transpose:
      CheckArrayShape()
      return CopyImage(array)
    else:
      CheckArrayShape()
      # copy image to new buffer, copy buffer.T to array
      array[:] = CopyImage(MakeBuffer()).T
      return array
  else:
    if not transpose:
      return CopyImage(MakeBuffer())
    else:
      return CopyImage(MakeBuffer()).T
  assert False, "Internal logic error!"

def ShowImage(img, fname = None):
  """Display an image to the user."""
  if sys.platform == "darwin":
    img.show()
  else:
    ShowImageOnLinux(img, fname)

def ShowImageOnLinux(img, fname = None):
  """Display an image to the user under Linux."""
  dir = TempDir()
  if not fname or '..' in fname:
    fname = 'img.png'
  path = dir.MakePath(fname)
  img.save(path)
  RunCommand("eog -n %s" % path, False, False)

def PowerSpectrum2d(image, width = None):
  """Compute the 2-D power spectrum for an image.

  :param image: Image data.
  :type image: 2D ndarray
  :param int width: Width of image to use for FFT (i.e., image width plus
     padding). By default, this is the width of the image.
  :returns: Squared amplitude from FFT of image.
  :rtype: 2D ndarray

  """
  if width != None:
    image = PadArray(image,
        np.repeat(width, 2),  # shape of padded array
        0)  # border value
  from scipy.fftpack import fftshift, fft2
  return np.abs(fftshift(fft2(image))) ** 2

def PowerSpectrum(image, width = None):
  """Get the 1-D power spectrum (squared-amplitude at each frequency) for a
  given input image. This is computed from the 2-D power spectrum via a
  rotational average.

  :param image: Image data.
  :type image: 2D ndarray
  :param int width: Width of image to use for FFT (i.e., image width plus
     padding). By default, this is the width of the image.
  :returns: Array whose rows contain the value, sum, and count of bins in the
     power histogram.

  """
  # from: http://www.astrobetter.com/fourier-transforms-of-images-in-python/
  assert image.ndim == 2
  f2d = PowerSpectrum2d(image, width)
  # Get sorted radii.
  x, y = np.indices(f2d.shape)
  center_x = (x.max() - x.min()) / 2.0
  center_y = (y.max() - y.min()) / 2.0
  r = np.hypot(x - center_x, y - center_y)
  ind = np.argsort(r.flat)
  r_sorted = r.flat[ind]
  # Bin the radii based on integer values. First, find the location (offset) for
  # the edge of each bin.
  r_int = r_sorted.astype(int)
  delta_r = r_int[1:] - r_int[:-1]
  r_ind = np.where(delta_r)[0]
  # Compute the number of elements in each bin.
  size_per_bin = r_ind[1:] - r_ind[:-1]
  # Finally, compute the average value for each bin.
  f_sorted = f2d.flat[ind]
  f_cumsum = np.cumsum(f_sorted, dtype = float)  # total cumulative sum
  sum_per_bin = f_cumsum[r_ind[1:]] - f_cumsum[r_ind[:-1]]  # cum. sum per bin
  # Use a circular window
  size = min(f2d.shape)
  sum_per_bin = sum_per_bin[: size / 2]
  size_per_bin = size_per_bin[: size / 2]
  # Compute the frequency (in cycles per pixel) corresponding to each bin.
  freq = np.arange(0, size / 2).astype(float) / size
  # Compute the average power for each bin.
  # XXX the average may be significantly more accurate than the sum, as there
  # are many fewer low-frequency locations in the FFT.
  #~ avg_per_bin = sum_per_bin / size_per_bin
  return np.array([freq, sum_per_bin, size_per_bin])

def MakeScalePyramid(data, num_layers, scale_factor):
  """Create a pyramid of resized copies of a 2D array.

  :param data: Base layer of the scale pyramid (i.e., highest frequency data).
  :type data: 2D ndarray of float
  :param int num_layers: Total number of layers in the pyramid, including
     the first layer passed as an argument.
  :param float scale_factor: Down-sampling factor between layers. Must be less
     than 1.
  :return: All layers of the scale pyramid.
  :rtype: list of 2D ndarray of float

  """
  if scale_factor >= 1:
    raise ValueError("Scale factor must be less than one.")
  pyramid = [ data ]
  image = toimage(data, mode = 'F')
  for i in range(num_layers - 1):
    size = np.array(image.size, np.int) * scale_factor
    image = image.resize(np.round(size).astype(int), Image.ANTIALIAS)
    pyramid.append(fromimage(image))
  return pyramid
