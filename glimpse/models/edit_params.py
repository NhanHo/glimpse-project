#!/usr/bin/env python

# Copyright (c) 2011 Mick Thomure
# All rights reserved.
#
# Please see the file COPYING in this distribution for usage
# terms.

# Read and write glimpse parameter files using a GUI.

from glimpse import glab, util
import sys

def main():
  try:
    model_class = glab.GetModelClass()
    ofile = None
    params = None
    cli_opts, cli_args = util.GetOptions('i:m:o:')
    for opt, arg in cli_opts:
      if opt == '-i':
        params = util.Load(arg)
      elif opt == '-m':
        model_class = glab.CLIGetModel(arg)
      elif opt == '-o':
        ofile = arg
  except util.UsageException, e:
    util.Usage("[options]\n"
        "  -i FILE              Read params from FILE\n"
        "  -m MODEL             Use model class MODEL\n"
        "  -o FILE              Write params to FILE",
        e
    )
  if params == None:
    params = model_class.ParamClass()
  if ofile == None:
    ofile = sys.stdout
  params.configure_traits()
  util.Store(params, ofile)

if __name__ == '__main__':
  main()
