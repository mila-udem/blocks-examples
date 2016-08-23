"""Encoder-Decoder with search for machine translation.

In this demo an encoder-decoder architecture with an attention mechanism is used for
machine translation. The attention mechanism is implemented according to
[BCB]_. The training data used is WMT15 Czech to English corpus, which you have
to download, preprocess and put to your 'datadir' in the config file. Note
that you can use `prepare_data.py` script to download and apply all the
preprocessing steps needed automatically.  Please see `prepare_data.py` for
further options of preprocessing.

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

import argparse
import logging
import pprint

import configurations

from machine_translation import main
from machine_translation.stream import get_tr_stream, get_dev_stream

logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_cs2en",
                    help="Prototype config to use for config")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")
args = parser.parse_args()


if __name__ == "__main__":
    # Get configurations for model
    configuration = getattr(configurations, args.proto)()
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    # Get data streams and call main
    main(configuration, get_tr_stream(**configuration),
         get_dev_stream(**configuration), args.bokeh)
