"""Utility classes and methods"""

import os
import logging
import queue
import shutil
import numpy as np
import torch
import torch.utils.data as data
import re

from skimage import io
from PIL import Image
from collections import Counter
from tqdm import tqdm

class NEWS(data.Dataset):
    """NEWS Dataset.
    
    Each item in the dataset is a tuple with the following entries (in order):
        - input_idxs: Indices of the tokens in title and content.
            Shape (input_len,).
        - atten_masks: Masks of input indices.
            Shape (input_len,).
        - img: Image of NEWS.
            Shape (x, x, 3).
        - y: Category index of NEWS
    
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset and image paths.
        transform (obj): Transform to apply on image.
    """

    def __init__(self, data_path, transform):
        super(NEWS, self).__init__()

        dataset = np.load(data_path, allow_pickle=True)
        self.input_idxs = torch.from_numpy(dataset['input_idxs']).long()
        self.atten_masks = torch.from_numpy(dataset['atten_masks']).long()
        # self.ids = torch.from_numpy(dataset['ids']).long()
        self.y = torch.from_numpy(dataset['y']).long()
        self.transform = transform
    
    def __getitem__(self, idx):
        # image = io.imread(self.img_paths[idx])
        # image = Image.fromarray(image)
        # image = self.transform(image)

        example = (self.input_idxs[idx],
                   self.atten_masks[idx],
                   self.y[idx])
        return example

    def __len__(self):
        return len(self.y)

def collate_fn(examples):
    """ist of individCreate batch tensors from a lual examples returned
    by `NEWS.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (input_idxs, atten_masks, images, y)

    Returns:
        examples (tuple): Tuple of tensors (input_idxs, atten_masks, images, y).
        All of shape (batch_size, ...)
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.stack(scalars).float()

    def merge_input(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, max(lengths)
    
    def merge_mask(arrays, length, dtype=torch.int64):
        merged = torch.zeros(len(arrays), length, dtype=dtype)
        for i, seq in enumerate(arrays):
            merged[i] = seq[:length]
        return merged
    
    def merge_image(arrays, dtype=torch.float32):
        channel, height, weight = arrays[0].size()
        merged = torch.zeros(len(arrays), channel, height, weight, dtype=dtype)
        for i, image in enumerate(arrays):
            merged[i] = image
        return merged
    
    # Group by tensor type
    input_idxs, atten_masks, y = zip(*examples)
    # Merge into batch tensors
    input_idxs, length = merge_input(input_idxs)
    atten_masks = merge_mask(atten_masks, length)
    # images = merge_image(images)
    # ids = merge_0d(ids)
    y = merge_0d(y)
    # y = y

    return (input_idxs, atten_masks, y)

def get_pred_ans_pair(ids: list, probs: list, y: list):
    """Pair the prediction and the ground truth label

    Args:
        ids (list): List of NEWS IDs.
        probs (list): List of probability prediction.
        y (list): List of ground truth label.
    
    Return:
        pred_dict (dict): Dictionary index IDs -> (predicted label, true label)
    """
    pred_dict = {}
    for nid, prob, label in zip(ids, probs, y):
        # pred = prob.index(max(prob))
        for j in range(len(prob)):
            if int(prob[j]) >= 0.5:
                prob[j] = 1
            else:
                prob[j] = 0
        pred_dict[nid] = (prob, label)
        
    return pred_dict

def compute_f1(pred_dict):
    
    pred_counter = Counter()
    label_counter = Counter()
    correct_counter = Counter()

    for value in pred_dict.values():
        pred = [str(x) for x in value[0]]
        label = [re.sub('.0','',str(x)) for x in value[1]]
        for i in range(len(pred)):
            pred_counter[i] += 1
            label_counter[i] += 1
            correct_counter[i] += (pred[i] == label[i])
    f1_counter = Counter()
    f2_counter = Counter()
    f3_counter = Counter()
    f4_counter = Counter()
    for key in label_counter.keys():
        if correct_counter[key] == 0 :
            f1_counter[key] = 0
            continue
        if key == 0:
            precision = correct_counter[key] / pred_counter[key]
            recall = correct_counter[key] / label_counter[key]
            f1_counter[key] = (2 * precision * recall) / (precision + recall)
        elif key == 1:
            precision = correct_counter[key] / pred_counter[key]
            recall = correct_counter[key] / label_counter[key]
            f2_counter[key] = (2 * precision * recall) / (precision + recall)
        elif key == 2:
            precision = correct_counter[key] / pred_counter[key]
            recall = correct_counter[key] / label_counter[key]
            f3_counter[key] = (2 * precision * recall) / (precision + recall)
        elif key == 3:
            precision = correct_counter[key] / pred_counter[key]
            recall = correct_counter[key] / label_counter[key]
            f4_counter[key] = (2 * precision * recall) / (precision + recall)

    f1 = sum(f1_counter.values()) / len(f1_counter) * 100.
    f2 = sum(f2_counter.values()) / len(f2_counter) * 100.
    f3 = sum(f3_counter.values()) / len(f3_counter) * 100.
    f4 = sum(f4_counter.values()) / len(f4_counter) * 100.
    return f1, f2, f3, f4

# Credit to Chris Chute (chute@stanford.edu)
# (https://github.com/minggg/squad/blob/master/util.py)
def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')

# Credit to Chris Chute (chute@stanford.edu)
# (https://github.com/minggg/squad/blob/master/util.py)
def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Credit to Chris Chute (chute@stanford.edu)
# (https://github.com/minggg/squad/blob/master/util.py)
def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

# Credit to Chris Chute (chute@stanford.edu)
# (https://github.com/minggg/squad/blob/master/util.py)
def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

# Credit to Chris Chute (chute@stanford.edu)
# (https://github.com/minggg/squad/blob/master/util.py)
class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

# Credit to Chris Chute (chute@stanford.edu)
# (https://github.com/minggg/squad/blob/master/util.py)
class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count