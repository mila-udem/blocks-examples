from __future__ import print_function

import logging
import numpy
import operator
import os
import re
import signal
import time


from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)


#=====================================================================#


def draw_align_map(attend_weights, input_length, sample_length, save_path,
                   input_words=None, predict_words=None, target_sent=None):

    data = attend_weights.T[:input_length, :sample_length]

    fix, ax = pyplot.subplots()
    heatmap = ax.pcolor(data, cmap=pyplot.cm.Blues, alpha=0.8)

    fig = pyplot.gcf()
    fig.set_size_inches(16, 22)

    if target_sent:
        pyplot.title('Target: ' + target_sent.decode('utf_8'))

    ax.set_frame_on(False)

    ax.set_xlim([0, data.shape[1]])
    ax.set_ylim([0, data.shape[0]])
    ax.set_yticks(numpy.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_xticks(numpy.arange(data.shape[1]) + 0.5, minor=False)

    # ax.invert_yaxis()

    if predict_words:
        xlabels = map(lambda w: w.decode('utf8'), predict_words)
    else:
        xlabels = ['word_{}'.format(i) for i in range(data.shape[1])]

    if input_words:
        ylabels = map(lambda w: w.decode('utf8'), input_words)
    else:
        ylabels = ['word_{}'.format(i) for i in range(data.shape[0])]

    ax.set_xlabel('output sentence')
    ax.set_xticklabels(xlabels, minor=False)
    ax.set_ylabel('input sentence')
    ax.set_yticklabels(ylabels, minor=False)

    pyplot.xticks(rotation=90)

    ax.grid(False)

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    pp = PdfPages(save_path)
    pyplot.savefig(pp, format='pdf')
    pp.close()

#=====================================================================#


class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, save_to, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()
        self.save_to = save_to

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            _1, outputs, _2, attend_weight, costs = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)
            input_sent = self._idx_to_word(input_[i][:input_length], self.src_ivocab)
            sample_sent = self._idx_to_word(outputs[:sample_length], self.trg_ivocab)
            target_sent = self._idx_to_word(target_[i][:target_length], self.trg_ivocab)

            print("Input : ", input_sent)
            print("Target: ", target_sent)
            print("Sample: ", sample_sent)
            print("Sample cost: ", costs[:sample_length].sum())
            print()

            n_iter = self.main_loop.status['iterations_done']
            save_dir = os.path.join(self.save_to, 'iter{}'.format(n_iter))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            heatmap_path = os.path.join(save_dir, 'sample{}_heatmap.pdf'.format(i))
            print ('Drawing align heatmap for [sample{}] in [{}]'
                   .format(i, heatmap_path))
            draw_align_map(numpy.squeeze(attend_weight), input_length, sample_length,
                           heatmap_path, input_sent.split(), sample_sent.split(), target_sent)


class BleuValidator(SimpleExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_grndtruth'], '<']

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                        'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        # Get target vocabulary
        sources = self._get_attr_rec(self.main_loop, 'data_stream')
        trg_vocab = sources.data_streams[1].dataset.dictionary
        self.trg_ivocab = {v: k for k, v in trg_vocab.items()}
        trg_eos_sym = sources.data_streams[1].dataset.eos_token
        self.trg_eos_idx = trg_vocab[trg_eos_sym]

        if self.verbose:
            ftrans = open(self.config['val_set_out'], 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            input_ = numpy.tile(seq, (self.config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            numpy.savez(
                model.path, **self.main_loop.model.get_parameter_dict())
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_bleu_model_%d_BLEU%.2f.npz' %
            (int(time.time()), self.bleu_score) if path else None)
        return gen_path
