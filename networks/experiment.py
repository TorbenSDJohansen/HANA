# -*- coding: utf-8 -*-
"""
@author: tsdj

Contains the code in which the experiment is conducted.
"""

import os
import types
import json
import functools

import torch

from torch.utils.tensorboard import SummaryWriter

from networks.util.pytorch_functions import split_data

def _rgetattr(obj, attr, *args):
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    # Simple way to retrieve attributes of attributes (i.e. solve the problem
    # of nested objects). Useful to freeze (and unfreeze) layers that consists
    # of other layers.
    # This way, a PART of a submodule can be frozen. For example, if a module
    # contains a submodule for feature extraction, a submodule of the feature
    # extraction submodule can be frozen (rather than only the entire feature
    # extraction submodule)
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class NetworkExperimentSequences():
    ''' Class to perform experiments on networks, to give indications of
    performance between models, hyperparameters, optimization
    strategies, etc. Assumes the networks takes as inputs an image and outputs
    a probability distribution over a sequence (such as in SVHN).

    How to use:
        Initialize an instance as desired. Then, use the method self.run().
    '''

    def __init__(
            self,
            dataset: torch.utils.data.dataloader.DataLoader,
            models: dict,
            loss_function: types.FunctionType,
            eval_loss: types.FunctionType,
            acc_function: types.FunctionType,
            epochs: int,
            log_interval: int,
            save_interval: int,
            root: str,
            device: torch.device, # pylint: disable=E1101
        ):
        ''' Initializes the parameters of the network experiment.
        params:
            :: dataset: the torch data set training data is drawn from.
            :: models: a dictionary of the models, including optimizer,
               scheduler, info on state_objects, and step_objects.
            :: loss_function: function that takes as inputs output, target
               and returns a loss.
            :: eval_loss: function used to track loss of validation data,
               returning both the sequence loss (int) and individual digit
               losses (list of ints).
            :: acc_function: function that takes as inputs output, target
               and returns a tuple of two elements, the first of which is
               the overall accuracy of the sequence and the second is a list
               of the individual accuracy for each element in the sequence.
            :: epochs: the number of epochs to train each network. Note that
               if 7 is given but a network has already been trained for 5
               epochs, it is trained for an additional 7, resulting in 12
               total epochs of training.
            :: log_inverval: how often to log stats to tensorboard (in steps).
            :: save_interval: how often to save model (in steps).
            :: root: the "root" folder in which the subfolders with logs
               are located.
            :: device: whether to run on CPU or GPU.
        '''

        assert (
            isinstance(dataset, torch.utils.data.dataloader.DataLoader)
            and isinstance(models, dict)
            and isinstance(loss_function, types.FunctionType)
            and isinstance(acc_function, types.FunctionType)
            and isinstance(eval_loss, types.FunctionType)
            and isinstance(epochs, int)
            and epochs > 0
            and isinstance(log_interval, int)
            and log_interval > 0
            and isinstance(save_interval, int)
            and save_interval > 0
            and isinstance(device, torch.device) # pylint: disable=E1101
            )

        self.dataset = dataset
        self.models = models
        self.loss_function = loss_function
        self.acc_function = acc_function
        self.eval_loss = eval_loss # if eval_loss is not None else loss_function

        self.epochs = epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.root = root
        self.device = device

        self.log_folders = {x: f'{root}/logs/{x}/' for x in models.keys()}

        self._check_folders()
        self._check_objects()

        self.info = None
        self.test_split = None
        self.nb_test_obs = None
        self.running_losses = None
        self.log_test_results = None

    def _check_folders(self):
        '''Verifies that the needed folders for logs and saves exist. If not,
        makes them.
        '''
        for key in self.log_folders:
            if not os.path.exists(self.log_folders[key]):
                print(f'Folder {self.log_folders[key]} missing! Making it.')
                os.makedirs(self.log_folders[key])

    def _check_objects(self):
        ''' Verifies that if a model, for example, specifies that the
        scheduler is an object on which .step() should be performed that a
        scheduler actually exists.
        '''
        for model_name in self.models:
            if 'state_objects' in self.models[model_name].keys():
                for key in self.models[model_name]['state_objects']:
                    assert key in self.models[model_name].keys()
            if 'step_objects' in self.models[model_name].keys():
                for key in self.models[model_name]['step_objects']:
                    assert key in self.models[model_name].keys()

    def _save_model(self, model_name):
        ''' Saves all things related to model, including (if relevant)
        the optimizer's and scheduler's state. Also some meta information,
        so we know the current epoch and the current step.
        '''
        # object holds a subset of the keys in self.models[model_name],
        # specifically those that have an associates state_dict, such as
        # the model, the optimizer, and the scheduler. This way, everything
        # that shall be saved thus can be saved thus.
        state_objects = self.models[model_name]['state_objects']

        epoch = self.info[model_name]['epoch']
        step = self.info[model_name]['step']

        meta_info = {
            'state_objects': state_objects,
            'epoch': epoch,
            'step': step,
            }

        for key in state_objects:
            file = f'{self.log_folders[model_name]}{key}_{step}.pt'
            torch.save(self.models[model_name][key].state_dict(), file)

        json.dump(
            meta_info,
            open(f'{self.log_folders[model_name]}meta_info.json', 'w'),
            )

    def _load_model(self, model_name):
        ''' Loads all things related to model, including (if relevant)
        the optimizer's and scheduler's state. Starts from the newest epoch
        where all information is saved (which fixes the case of a crash
        occurring during a save that results in a model's state being saved
        but an optimizer's state not being saved.)
        '''
        file_meta_info = f'{self.log_folders[model_name]}meta_info.json'
        if os.path.isfile(file_meta_info):
            print(f'Loading {model_name}.')
            meta_info = json.load(open(file_meta_info, 'rb'))

            state_objects = meta_info['state_objects']
            epoch = meta_info['epoch']
            step = meta_info['step']

            for key in state_objects:
                file = f'{self.log_folders[model_name]}{key}_{step}.pt'
                self.models[model_name][key].load_state_dict(torch.load(file))
        else:
            print(f'Nothing to load for {model_name}; initializing.')
            epoch, step = 0, 0 # no training done yet; start from zero.

        return epoch, step

    def _setup_tensorboard(self, model_name, x_train, step):
        ''' Setup a tensorboard writer for each model, including creating
        a graph.
        '''
        self.models[model_name]['model'].cpu()
        self.models[model_name]['model'].eval()

        writer = SummaryWriter(log_dir=self.log_folders[model_name])

        if step == 0:
            writer.add_graph( # NOTE: This is potentially VERY slow (due to cpu). Also, test if `with torch.no_grad` is fine here
                self.models[model_name]['model'],
                input_to_model=(x_train.float()),
                verbose=False,
                )
        writer.flush()

        self.models[model_name]['model'].to(self.device)

        return writer

    def _freeze_layers(self, model_name):
        if 'to_freeze' in self.models[model_name].keys():
            for layer in self.models[model_name]['to_freeze']:
                print(f'Freezing {layer}!')
                params = _rgetattr(self.models[model_name]['model'], layer).parameters()
                for param in params:
                    param.requires_grad = False

    def _unfreeze_layers(self, model_name: str, step: int):
        if 'to_unfreeze' in self.models[model_name].keys():
            for layer, when in self.models[model_name]['to_unfreeze']:
                if when == step:
                    print(f'Unfreezing {layer} at step {step}!')
                    params = _rgetattr(self.models[model_name]['model'], layer).parameters()
                    for param in params:
                        param.requires_grad = True

    def _setup(self):
        ''' Setup the models, such as retrieving information on current
        epoch and step as well as set up writers.
        '''
        data = next(iter(self.dataset))
        x_train = data['image'][:1]
        self.info = dict()

        for model_name in self.models.keys():
            epoch, step = self._load_model(model_name)
            writer = self._setup_tensorboard(model_name, x_train, step)

            self.info[model_name] = {
                'epoch': epoch,
                'step': step,
                'writer': writer,
                }

            self._freeze_layers(model_name)
            self.models[model_name]['model'].train()

    def _write_loss_tb(self, model_name, step, prefix, seq_loss, losses):
        ''' Writes a loss (a scalar) to tensorboard for specific model. '''
        if losses:
            self.info[model_name]['writer'].add_scalar(
                f'{prefix}/Losses/Loss', seq_loss, step,
                )
            for i, loss in enumerate(losses):
                self.info[model_name]['writer'].add_scalar(
                    f'{prefix}/Losses/"Digit" {i} loss', loss, step,
                    )
        else:
            self.info[model_name]['writer'].add_scalar(f'{prefix}/Losses/Loss', seq_loss, step)

    def _write_acc_tb(self, model_name, step, prefix, seq_acc, accs):
        ''' Writes accuracies (scalars) to tensorboard for specific model. '''
        self.info[model_name]['writer'].add_scalar(
            f'{prefix}/Accuracies/Accuracy', seq_acc, step,
            )
        for i, acc in enumerate(accs):
            self.info[model_name]['writer'].add_scalar(
                f'{prefix}/Accuracies/"Digit" {i} accuracy', acc, step,
                )

    def _write_tb(self, model_name, step, prefix, seq_acc, accs, seq_loss, losses=None):
        self._write_loss_tb(model_name, step, prefix, seq_loss, losses)
        self._write_acc_tb(model_name, step, prefix, seq_acc, accs)

    def _train_step(self, model_name, x_train, y_train):
        ''' Performs a single training step for a sigle model, i.e.
        forward -> loss -> backward -> steps.
        '''
        self.models[model_name]['optimizer'].zero_grad()

        yhat = self.models[model_name]['model'](x_train)
        loss = self.loss_function(yhat=yhat, y=y_train)
        loss.backward()

        for step_object in self.models[model_name]['step_objects']:
            self.models[model_name][step_object].step()

        return loss.item()

    def _print_run_info(self):
        print(
            f'''
Training {len(self.models)} models for {self.epochs} epochs,
each epoch lasting {len(self.dataset)} steps with batch size
{self.dataset.batch_size}.

Directory: {self.root}.

Logging every {self.log_interval} step.
Logging test results: {self.log_test_results}.

Saving every {self.save_interval} step.
            ''',
            )

    @torch.no_grad()
    def _log(self, model_name, x_train, y_train):
        self.models[model_name]['model'].eval()

        # with torch.no_grad():
        yhat = self.models[model_name]['model'](x_train)
        seq_acc, accs = self.acc_function(yhat=yhat, y=y_train)
        self._write_tb(
            model_name=model_name,
            step=self.info[model_name]['step'],
            prefix='Train',
            seq_acc=seq_acc,
            accs=accs,
            seq_loss=self.running_losses[model_name],
            )

        if self.log_test_results:
            seq_loss_test_list = []
            losses_test_list = []
            seq_acc_test_list = []
            accs_test_list = []

            for x_test_batch, y_test_batch in self.test_split:
                batch_size = len(x_test_batch)

                yhat_test_batch = self.models[model_name]['model'](
                    x_test_batch.to(self.device).float()
                    )
                seq_loss_test_batch, losses_test_batch = self.eval_loss(
                    yhat=yhat_test_batch, y=y_test_batch.to(self.device).long()
                    )
                seq_acc_test_batch, accs_test_batch = self.acc_function(
                    yhat=yhat_test_batch, y=y_test_batch.to(self.device).long())

                seq_loss_test_list.append(batch_size * seq_loss_test_batch)
                losses_test_list.append([x * batch_size for x in losses_test_batch])

                seq_acc_test_list.append(batch_size * seq_acc_test_batch)
                accs_test_list.append([x * batch_size for x in accs_test_batch])

            seq_loss_test = sum(seq_loss_test_list) / self.nb_test_obs
            losses_test = [
                sum([x[i] for x in losses_test_list]) / self.nb_test_obs
                for i in range(len(yhat_test_batch))
                ]

            seq_acc_test = sum(seq_acc_test_list) / self.nb_test_obs
            accs_test = [
                sum([x[i] for x in accs_test_list]) / self.nb_test_obs
                for i in range(len(yhat_test_batch))
                ]

            self._write_tb(
                model_name=model_name,
                step=self.info[model_name]['step'],
                prefix='Test',
                seq_acc=seq_acc_test,
                accs=accs_test,
                seq_loss=seq_loss_test,
                losses=losses_test,
                )

        self.models[model_name]['model'].train()

    def _run_model_step(self, model_name, x_train, y_train, step):
        loss = self._train_step(model_name, x_train, y_train)

        self.running_losses[model_name] = (
            (self.running_losses[model_name] * (step - 1) + loss) / step
            )
        self.info[model_name]['step'] += 1

        # TODO maybe also always write on step == 1 (useful to check start!).
        # Or even write on step 0 - i.e. before first train step
        if self.info[model_name]['step'] % self.log_interval == 0:
            self._log(model_name, x_train, y_train)

        if self.info[model_name]['step'] % self.save_interval == 0:
            self._save_model(model_name)

        self._unfreeze_layers(model_name, self.info[model_name]['step'])

    def run(self, test_data=None):
        ''' Only "public" method, which runs the experiment. If test data is
        provided, it logs the test losses.
        Training losses are averaged over the epoch and always logged.
        '''
        self.log_test_results = test_data is not None

        if self.log_test_results:
            self.test_split, self.nb_test_obs = split_data(
                data=test_data,
                batch_size=self.dataset.batch_size,
                )
            del test_data

        self._print_run_info()
        self._setup()

        for epoch in range(self.epochs):
            try:
                print(f'Starting epoch {epoch + 1} of {self.epochs}')
                self.running_losses = {model_name: 0 for model_name in self.models.keys()}

                for step, data in enumerate(self.dataset, start=1):
                    x_train = data['image'].to(self.device).float()
                    y_train = data['label'].to(self.device).long()

                    for model_name in self.models.keys():
                        self._run_model_step(model_name, x_train, y_train, step)

                for model_name in self.models.keys():
                    self.info[model_name]['epoch'] += 1
            except KeyboardInterrupt:
                break

        for model_name in self.models.keys():
            self._save_model(model_name)
