# Encoder-Decoder with search for machine translation.
In this demo an encoder-decoder architecture with an attention mechanism is used for
machine translation. The attention mechanism is implemented according to
[1]. The training data used is WMT15 Czech to English corpus, which you have
to download, preprocess and put to your `datadir` in the config file. Note
that you can use `prepare_data.py` script to download and apply all the
preprocessing steps needed automatically.  Please see `prepare_data.py` for
further options of preprocessing.

[1] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.

# Usage:

From the `./blocks-examples` root type in the in the command line:

```
python machine_translation/prepare_data.py
python -m machine_translation
```

# Expected results:
Using an NVIDIA Tesla K80 gpu we got the following results after 1 hour and 1200 iterations, on the first epoch.

| Measure  | Score |
| -------- |------ |
| Cost  | 123.59  |
| Bleu  | XXX  |

# Additional Notes:
There is no support for CPU-only platforms at the moment. Make sure you are using a CUDA-enabled GPU with ~100MB of free space. 
