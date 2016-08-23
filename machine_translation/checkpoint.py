
import logging
import numpy
import os
import time

from contextlib import closing
from six.moves import cPickle

from blocks.extensions.saveload import SAVED_TO, LOADED_FROM
from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from blocks.utils import reraise_as

logger = logging.getLogger(__name__)


class SaveLoadUtils(object):
    """Utility class for checkpointing."""

    @property
    def path_to_folder(self):
        return self.folder

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        return os.path.join(self.folder, 'log')

    def load_parameter_values(self, path):
        with closing(numpy.load(path)) as source:
            param_values = {}
            for name, value in source.items():
                if name != 'pkl':
                    name_ = name.replace(BRICK_DELIMITER, '/')
                    if not name_.startswith('/'):
                        name_ = '/' + name_
                    param_values[name_] = value
        return param_values

    def save_parameter_values(self, param_values, path):
        param_values = {name.replace("/", BRICK_DELIMITER): param
                        for name, param in param_values.items()}
        numpy.savez(path, **param_values)

    def set_model_parameters(self, model, params):
        params_this = model.get_parameter_dict()
        missing = set(params_this.keys()) - set(params.keys())
        for pname in params_this.keys():
            if pname in params:
                val = params[pname]
                if params_this[pname].get_value().shape != val.shape:
                    logger.warning(
                        " Dimension mismatch {}-{} for {}"
                        .format(params_this[pname].get_value().shape,
                                val.shape, pname))

                params_this[pname].set_value(val)
                logger.info(" Loaded to CG {:15}: {}"
                            .format(val.shape, pname))
            else:
                logger.warning(
                    " Parameter does not exist: {}".format(pname))
        logger.info(
            " Number of parameters loaded for computation graph: {}"
            .format(len(params_this) - len(missing)))


class CheckpointNMT(SimpleExtension, SaveLoadUtils):
    """Redefines checkpointing for NMT.

        Saves only parameters (npz), iteration state (pickle) and log (pickle).

    """

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        kwargs.setdefault("after_training", True)
        super(CheckpointNMT, self).__init__(**kwargs)

    def dump_parameters(self, main_loop):
        params_to_save = main_loop.model.get_parameter_values()
        self.save_parameter_values(params_to_save,
                                   self.path_to_parameters)

    def dump_iteration_state(self, main_loop):
        secure_dump(main_loop.iteration_state, self.path_to_iteration_state)

    def dump_log(self, main_loop):
        secure_dump(main_loop.log, self.path_to_log, cPickle.dump)

    def dump(self, main_loop):
        if not os.path.exists(self.path_to_folder):
            os.mkdir(self.path_to_folder)
        print("")
        logger.info(" Saving model")
        start = time.time()
        logger.info(" ...saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" ...saving iteration state")
        self.dump_iteration_state(main_loop)
        logger.info(" ...saving log")
        self.dump_log(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

    def do(self, callback_name, *args):
        try:
            self.dump(self.main_loop)
        except Exception:
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (self.path_to_folder +
                                                            'params.npz',))


class LoadNMT(TrainingExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadNMT, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.path_to_folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.path_to_folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path_to_folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            self.set_model_parameters(main_loop.model, params_all)
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
