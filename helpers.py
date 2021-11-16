from argparse import Namespace
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
def column_gather(y_out, x_lengths):
    '''Get a specific vector from each batch datapoint in `y_out`.

    More precisely, iterate over batch row indices, get the vector that's at
    the position indicated by the corresponding value in `x_lengths` at the row
    index.

    Args:
        y_out (torch.FloatTensor, torch.cuda.FloatTensor)
            shape: (batch, sequence, feature)
        x_lengths (torch.LongTensor, torch.cuda.LongTensor)
            shape: (batch,)

    Returns:
        y_out (torch.FloatTensor, torch.cuda.FloatTensor)
            shape: (batch, feature)
    '''
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1

    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])

    return torch.stack(out)

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings 
    
    Args:
        glove_filepath (str): path to the glove embeddings file 
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, encoding="utf8") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line: word num1 num2 ...
            word_to_index[line[0]] = index # word = line[0] 
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.
    
    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    
    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings


# def make_train_state(args):
#     return {'stop_early': False,
#             'early_stopping_step': 0,
#             'early_stopping_best_val': 1e8,
#             'learning_rate': args.learning_rate,
#             'epoch_index': 0,
#             'train_loss': [],
#             'train_acc': [],
#             'val_loss': [],
#             'val_acc': [],
#             'test_loss': -1,
#             'test_acc': -1,
#             'model_filename': args.model_state_file}


# def update_train_state(args, model, train_state):
#     """Handle the training state updates.

#     Components:
#      - Early Stopping: Prevent overfitting.
#      - Model Checkpoint: Model is saved if the model is better
    
#     :param args: main arguments
#     :param model: model to train
#     :param train_state: a dictionary representing the training state values
#     :returns:
#         a new train_state
#     """

#     # Save one model at least
#     if train_state['epoch_index'] == 0:
#         torch.save(model.state_dict(), train_state['model_filename'])
#         train_state['stop_early'] = False

#     # Save model if performance improved
#     elif train_state['epoch_index'] >= 1:
#         loss_tm1, loss_t = train_state['val_loss'][-2:]
         
#         # If loss worsened
#         if loss_t >= loss_tm1:
#             # Update step
#             train_state['early_stopping_step'] += 1
#         # Loss decreased
#         else:
#             # Save the best model
#             if loss_t < train_state['early_stopping_best_val']:
#                 torch.save(model.state_dict(), train_state['model_filename'])
#                 train_state['early_stopping_best_val'] = loss_t

#             # Reset early stopping step
#             train_state['early_stopping_step'] = 0

#         # Stop early ?
#         train_state['stop_early'] = \
#             train_state['early_stopping_step'] >= args.early_stopping_criteria

#     return train_state

# compute the loss & accuracy on the test set using the best available model

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
