#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os.path
import platform

import jax.random
import numpy as np
import optax
import torch
import torchvision.transforms
from flax import nnx
from torch.utils import data
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from densenet import DenseNet

seed = 12345
batch_size = 128

if platform.system() == 'Linux':
    num_train_workers = 8
    num_test_workers = 4
    persistent_workers = True
else:
    num_train_workers = 0
    num_test_workers = 0
    persistent_workers = False

dataset_path = os.path.realpath(os.path.curdir)

train_dataset = CIFAR10(root=dataset_path, download=True)
data_means = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
data_std = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
print("Data mean", data_means)
print("Data std", data_std)


def image_to_numpy(img, mean: float, std: float):
    """Transformations applied on each image => bring them into a numpy array"""
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - mean) / std
    return img


def numpy_collate(batch):
    """Stack the batch elements"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


test_transform = lambda img: image_to_numpy(img, data_means, data_std)
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    lambda img: image_to_numpy(img, data_means, data_std)
])

# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = CIFAR10(root=dataset_path, transform=train_transform)
train_set, _ = data.random_split(train_dataset, [45000, 5000],
                                 generator=torch.Generator().manual_seed(seed))
val_dataset = CIFAR10(root=dataset_path, transform=test_transform)
_, val_set = data.random_split(val_dataset, [45000, 5000],
                               generator=torch.Generator().manual_seed(seed))

# Loading the test set
test_set = CIFAR10(root=dataset_path, train=False, transform=test_transform)

# We define a set of data loaders that we can use for training and validation
train_loader = data.DataLoader(train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=num_train_workers,
                               persistent_workers=persistent_workers)
val_loader = data.DataLoader(val_set,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=numpy_collate,
                             num_workers=num_test_workers,
                             persistent_workers=persistent_workers)
test_loader = data.DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              collate_fn=numpy_collate,
                              num_workers=num_test_workers,
                              persistent_workers=persistent_workers)

imgs, _ = next(iter(train_loader))
print("Batch mean", imgs.mean(axis=(0, 1, 2)))
print("Batch std", imgs.std(axis=(0, 1, 2)))

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
)
metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
}


def loss_fn(model, batch):
    imgs, labels = batch
    # Run model. During training, we need to update the BatchNorm statistics.
    logits = model(imgs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric,
               batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])  # In-place updates.


def train_classifier(num_epochs=300):
    # Create a trainer module with specified hyperparameters
    print('create model')
    model = DenseNet(dims=2,
                     in_features=3,
                     num_classes=10,
                     num_blocks=(6, 6, 6, 6),
                     init_conv_kernel_size=3,
                     init_conv_strides=1,
                     bn_size=2,
                     growth_rate=16,
                     dropout_rate=0.2,
                     pre_dropout=True,
                     rngs=nnx.rnglib.Rngs(jax.random.PRNGKey(seed)))
    nnx.display(model)

    # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
    num_steps_per_epoch = len(train_loader)
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=1e-3,
        boundaries_and_scales={
            int(num_steps_per_epoch * num_epochs * 0.6): 0.1,
            int(num_steps_per_epoch * num_epochs * 0.85): 0.1}
    )
    optimizer = optax.chain(
        optax.clip(1.0),  # Clip gradients at 1
        optax.adamw(lr_schedule, weight_decay=1e-4)
    )
    optimizer = nnx.Optimizer(model, optimizer)  # reference sharing

    step = 0

    t1 = datetime.datetime.now()
    print(f'[{t1}] training begin')

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc='Training', leave=False):
            train_step(model, optimizer, metrics, batch)
            step += 1

        # One training epoch has passed, log the training metrics.
        for metric, value in metrics.compute().items():  # Compute the metrics.
            metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
        metrics.reset()  # Reset the metrics for the test set.

        # Compute the metrics on the test set after each training epoch.
        for test_batch in val_loader:
            eval_step(model, metrics, test_batch)

        # Log the test metrics.
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
        metrics.reset()  # Reset the metrics for the next training epoch.

        print(f'epoch {epoch}, step {step}:')
        print(f"  [train] "
              f"loss: {metrics_history['train_loss'][-1]}, "
              f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"  [test] "
              f"loss: {metrics_history['test_loss'][-1]}, "
              f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")

    t2 = datetime.datetime.now()
    print(f'[{t2}] training stop')
    print(f'Elapse {(t2 - t1).total_seconds()} seconds')

    return model


def eval_model(model, metrics, loader):
    for test_batch in loader:
        eval_step(model, metrics, test_batch)


model = train_classifier(num_epochs=300)

# Test trained model
metrics.reset()
eval_model(model, metrics, val_loader)
print('[validate]:')
for metric, value in metrics.compute().items():  # Compute the metrics.
    print(f'  {metric}={value}')
print('')

metrics.reset()
eval_model(model, metrics, test_loader)
print('[test]:')
for metric, value in metrics.compute().items():  # Compute the metrics.
    print(f'  {metric}={value}')
print('')
