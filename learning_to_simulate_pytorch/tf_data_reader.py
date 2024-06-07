import functools
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import reading_utils
import tree


def prepare_data_from_tfds(data_path='data/train.tfrecord', is_rollout=False, batch_size=2):
    metadata = reading_utils._read_metadata('data/')
    ds = tf.data.TFRecordDataset([data_path])
    ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if is_rollout:
        ds = ds.map(prepare_rollout_inputs)
    else:    
        split_with_window = functools.partial(
            reading_utils.split_trajectory,
            window_length=6 + 1)
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        ds = ds.repeat()
        ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    ds = tfds.as_numpy(ds)
    for i in range(100): # clear screen
        print()
    return ds


# from tfrecord.torch.dataset import TFRecordDataset
def prepare_inputs(tensor_dict):
    pos = tensor_dict['position']
    pos = tf.transpose(pos, perm=[1, 0, 2])
    target_position = pos[:, -1]
    tensor_dict['position'] = pos[:, :-1]
    num_particles = tf.shape(pos)[0]
    tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
    if 'step_context' in tensor_dict:
        tensor_dict['step_context'] = tensor_dict['step_context'][-2]
        tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
    return tensor_dict, target_position


def batch_concat(dataset, batch_size):
    windowed_ds = dataset.window(batch_size)
    initial_state = tree.map_structure(lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),dataset.element_spec)
    def reduce_window(initial_state, ds):
        return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
    return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))


def prepare_rollout_inputs(context, features):
    out_dict = {**context}
    pos = tf.transpose(features['position'], [1, 0, 2])
    target_position = pos[:, -1]
    out_dict['position'] = pos[:, :-1]
    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
    if 'step_context' in features:
        out_dict['step_context'] = features['step_context']
    out_dict['is_trajectory'] = tf.constant([True], tf.bool)
    return out_dict, target_position
