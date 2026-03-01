"""YAMNet fine-tuning training package.

Pipeline:
  1. dataset_curator (simulator repo) → labels.csv + WAV files
  2. data_loader.py                   → mel patches (numpy / tf.data)
  3. train_yamnet.py                  → fine-tuned Keras model checkpoint
  4. export_finetuned.py              → TFLite for ODAS + models/registry.json
"""
