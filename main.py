
import argparse

import tensoflow as tf
import tensorflow.keras as tfk

from .model import SiameseNet


parser=argparse.ArgumentParser(description="Siamese Net")

parser.add_argument(
    "--data_dir"
    ,type=str
    ,required=True
    ,help="Path to the dataset directory"
    )

args=parser.parse_args()

data_dir=args.data_dir

# Load model & data
model = SiameseNet()
model.loader(data_dir)

# Fit the model
model.fit(True,5)
