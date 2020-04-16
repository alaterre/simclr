import collections
from pathlib import Path
from typing import Iterable, Callable, Union, Sequence

import numpy as np
import tensorflow as tf
from absl import logging, flags, app
from matplotlib.image import imread
from tqdm import tqdm


flags.DEFINE_string("data_dir", None, "Dataset directory.")
FLAGS = flags.FLAGS

TF_RECORDS_COMPRESSION_TYPE = "GZIP"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: Union[Sequence[float], float]):
    """Returns a float_list from a float / double."""
    if isinstance(value, float):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: Union[Sequence[int], int]):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, int):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecords(
    pathname: Path,
    examples: Iterable,
    serializer: Callable,
    examples_per_tfrecord: int = 1000,
):
    options = tf.io.TFRecordOptions(compression_type=TF_RECORDS_COMPRESSION_TYPE)
    pathname.mkdir(parents=True, exist_ok=True)

    # Distribute the examples across the tfrecords.
    content = collections.defaultdict(list)
    for k, example in enumerate(examples):
        content[k // examples_per_tfrecord].append(example)

    # write the tf-records
    logging.info(f"Start writing tf-records, {len(content)} files to write.")
    for k, examples in tqdm(content.items(), desc="Writing tf-records... "):
        filename = str(pathname.joinpath(f"example_{k}.tfrecords"))
        with tf.io.TFRecordWriter(filename, options=options) as writer:
            for example in examples:
                writer.write(serializer(example))


def serialize_example(example):
    boardname, step, board = example
    feature_dict = {
        "shape": _int64_feature(board.shape),
        "image": _bytes_feature(board.tobytes()),
        "board_name": _bytes_feature(boardname.encode()),
        "step": _int64_feature(step),
    }
    tf_features = tf.train.Features(feature=feature_dict)
    example = tf.train.Example(features=tf_features)
    return example.SerializeToString()


def parser(example_proto):
    feature_dict = {
        "shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64),
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "board_name": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "step": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    parsed_features = tf.io.parse_single_example(
        serialized=example_proto, features=feature_dict
    )
    return parsed_features


def _convert_image(example):
    image, shape = example["image"], example["shape"]
    image = tf.io.decode_raw(image, out_type=tf.float32)
    image = tf.reshape(image, shape=shape)
    return image


def data_gen(boards):
    for boardname in boards:
        for step in boards[boardname]:
            yield boardname, step, boards[boardname][step]


"""
PLACEHOLDER
=============================

def read_dataset(
    pathname: str, batch_size, max_board_size, max_num_channels, padding_values
):
    files = tf.data.Dataset.list_files(f"{pathname}/*.tfrecords")
    dataset = files.interleave(
        lambda file: tf.data.TFRecordDataset(
            file, compression_type=TF_RECORDS_COMPRESSION_TYPE
        ),
        cycle_length=4,
        block_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).repeat()
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        _convert_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    gen_shapes = tf.TensorShape([max_board_size, max_board_size, max_num_channels])
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=gen_shapes, padding_values=padding_values
    )
"""


def main(_):
    data_path = Path.cwd().joinpath(FLAGS.data_dir)

    # read the dataset of images
    boards = collections.defaultdict(dict)
    for board_path in data_path.glob("[!.]*/"):
        logging.info(f"Reading board: {board_path.name}")

        content = collections.defaultdict(dict)
        for filename in board_path.glob("*.png"):
            # get info from filename
            step, layer = filename.name.split(".")[0].split("_")[1:]
            step, layer = int(step), int(layer)

            # read the data
            content[step][layer] = imread(str(filename))

        for step, layers in content.items():
            board = [layers[k] for k in sorted(layers.keys())]
            board = np.concatenate(board, axis=-1)
            boards[board_path.name][step] = board

    # write the tf_records
    gen = data_gen(boards)
    pathname = data_path.joinpath("tf_records")

    write_tfrecords(
        pathname=pathname,
        examples=gen,
        serializer=serialize_example,
        examples_per_tfrecord=1000,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    app.run(main)
