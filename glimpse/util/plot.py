
# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage terms.

#
# Functions for plotting data with Matplotlib.
#

import misc

import math
# Following imports assume matplotlib environment has been initialized.
from matplotlib import cm
from matplotlib import pyplot
from mpl_toolkits.axes_grid import AxesGrid
import numpy as np
import operator

# See glimpse.util.__init__.py for implementation of InitPlot().

def Histogram(data, nbins = 100, **args):
  """Create histogram of numpy ndarray. If output filename is given, write
  figure to file. Otherwise, display it on screen."""
  args = misc.MergeDict(args, normed = True, histtype = 'stepfilled')
  pyplot.hist(data.flat, nbins, **args)

_sp_y = _sp_x = _sp_j = 0
def MakeSubplot(y, x):
  global _sp_y, _sp_x, _sp_j
  _sp_y = y
  _sp_x = x
  _sp_j = 1

def NextSubplot(figure = None):
  global _sp_j
  if figure == None:
    figure = pyplot.gcf()
  figure.add_subplot(_sp_y, _sp_x, _sp_j)
  _sp_j += 1
  return figure.gca()

def Show2DArray(fg, bg = None, mapper = None, annotation = None, title = None,
    axes = None, show = True, colorbar = False, **fg_args):
  """Show a 2-D array using matplotlib.
  fg -- foreground array to display, shown with a spectral colormap
  bg -- (optional) background array, shown as grayscale
  mapper -- (required if bg is not None) function to map locations in fg to
            locations in bg
  annotation -- (optional) small 2-D array to display in top-left corner of plot
  title -- (optional) string to be displayed above plot
  """
  if axes == None:
    axes = pyplot.gca()
  if bg == None:
    f_extent = [0, fg.shape[-1], fg.shape[-2], 0]
    f_extent[1] += 1
    f_extent[2] += 1
    b_extent = f_extent
    falpha = 1
  else:
    assert mapper != None
    f_extent = map(mapper, (0, fg.shape[-1], fg.shape[-2], 0))
    f_extent[1] += 1
    f_extent[2] += 1
    falpha = 0.9
    b_extent = (0, bg.shape[-1], bg.shape[-2], 0)
  # Show smallest element first -- that means annotation, fg, bg
  hold = axes.ishold()
  if annotation != None:
    # Show annotation in upper left corner
    k_extent = (0, b_extent[1] * 0.2, b_extent[2] * 0.2, 0)
    axes.imshow(annotation, cmap = cm.gray, extent = k_extent,
        zorder = 3)
  axes.hold(True)
  # Show foreground
  axes.imshow(fg, **misc.MergeDict(fg_args, # cmap = cm.spectral,
      extent = f_extent, zorder = 2, alpha = falpha))
  if colorbar:
    axes.figure.colorbar(axes.images[-1], ax = axes)
  if bg != None:
    # Show base image in the background
    axes.imshow(bg, cmap = cm.gray, extent = b_extent, zorder = 1)
  if title != None:
    axes.set_title(title)
  # Leave the border around the image, but don't show any tick marks.
  axes.set_yticks([])
  axes.set_xticks([])
  axes.hold(hold)
  if show:
    axes.figure.show()

def ShowListWithCallback(xs, cb, cols = None, figure = None, nsubplots = None,
    colorbar = False):
  """Show each slice of a 3-D array using a callback function.
  xs -- list of values to pass to callback function. this could, e.g., be a 3-D
        array (which is a list of 2-D arrays), or a 1-D array of indices.
  cb -- callback function taking an axes object and the current dataset
  nsubplots -- number of total subplots. could be more then len(xs)
  """
  if figure == None:
    figure = pyplot.gcf()
  axes = figure.gca()
  figure.clf()  # necessary to avoid showing large axes in background
  max_plots = 64
  if len(xs) > max_plots:
    xs = xs[:max_plots]
  if nsubplots == None:
    nsubplots = len(xs)
  if cols == None:
    if nsubplots <= 12:
      cols = 4
    else:
      cols = 8
    cols_was_none = True
  else:
    cols_was_none = True
  rows = math.ceil(nsubplots / float(cols))
  if cols_was_none and rows == 1:
    cols = nsubplots
  MakeSubplot(rows, cols)
  for x in xs:
    NextSubplot(figure)
    cb(x)

def Show2DArrayList(xs, annotations = None, normalize = True, colorbar = False,
    colorbars = False, cols = None, center_zero = True, axes = None,
    figure = None, show = True, **args):
  """Display a list of 2-D arrays using matplotlib.
  annotations -- small images to show with each array visualization
  normalize -- use the same colormap range for all subplots
  colorbar -- show the meaning of the colormap as an extra subplot (implies
              normalize = True)
  colorbars -- show a different colorbar for each subplot
  cols -- number of subplot columns to use
  center_zero -- when normalizing a range that spans zero, make sure zero is in
                 the center of the colormap range
  """
  if 'vmin' in args and 'vmax' in args:
    vmin = args['vmin']
    vmax = args['vmax']
  elif colorbar or normalize:
    vmin = min([ x.min() for x in xs ])
    vmax = max([ x.max() for x in xs ])
    if center_zero and vmin < 0 and vmax > 0:
      vmax = max(abs(vmin), vmax)
      vmin = -vmax
    args = misc.MergeDict(args, vmin = vmin, vmax = vmax)
  if annotations == None:
    annotations = [None] * len(xs)
  else:
    assert len(annotations) == len(xs), \
        "Got %d arrays, but %d annotations (these numbers should match)" % \
        (len(xs), len(annotations))

  # Compute rows & cols
  max_plots = 64
  if len(xs) > max_plots:
    xs = xs[:max_plots]
  if cols == None:
    if len(xs) <= 16:
      cols = min(4, len(xs))
    else:
      cols = min(8, len(xs))
  rows = int(math.ceil(len(xs) / float(cols)))
  # Create the image grid
  if figure == None:
    figure = pyplot.gcf()
  figure.clf()
  if colorbar:
    grid_args = dict(cbar_location = "right", cbar_mode = "single")
  elif colorbars:
    grid_args = dict(cbar_location = "right", cbar_mode = "each",
        cbar_pad = "2%")
  else:
    grid_args = {}
  grid = AxesGrid(figure, 111,
    nrows_ncols = (rows, cols),
    axes_pad = 0.5,
    share_all = True,
    label_mode = "L",   # XXX value can be "L" or "1" -- what is this?
    **grid_args
  )
  # Add all subplots
  for i in range(len(xs)):
    Show2DArray(xs[i], annotation = annotations[i], axes = grid[i],
        show = False, **args)
  # Add the colorbar(s)
  if colorbar:
    img = grid[0].images[-1]
    grid.cbar_axes[0].colorbar(img)
    for cax in grid.cbar_axes:
      cax.toggle_label(True)
  elif colorbars:
    for i in range(len(xs)):
      img = grid[i].images[-1]
      ca = grid.cbar_axes[i]
      ca.colorbar(img)
      ca.toggle_label(True)
  if show:
    figure.show()

def Show3DArray(xs, annotations = None, figure = None, **args):
  """Display slices of a 3-D array using matplotlib.
  annotations -- small images to show with each array visualization
  """
  xs = xs.reshape((-1,) + xs.shape[-2:])
  if annotations != None:
    annotations = annotations.reshape((-1,) + annotations.shape[-2:])
  Show2DArrayList(xs, annotations, figure = figure, **args)