import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
import time

from data_loaders.dataset import CodeDataset
from models.astnn.model import BatchProgramClassifier
from torch.autograd import Variable

from models.model import CodeRepresentationModel


class ASTNN(CodeRepresentationModel):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.model = BatchProgramClassifier(self.EMBEDDING_DIM, params['dims']['hidden'], len(self.embedding),
                                            params['dims']['encode'], self.LABELS_SIZE, params['batch_size'],
                                            params['use_gpu'] == 1, self.embedding)
        if params['paths']['load_path']:
            self.load()
        else:
            if params['use_gpu']:
                self.model.cuda()
            parameters = self.model.parameters()
            self.optimizer = torch.optim.Adamax(parameters)
            self.loss_function = torch.nn.CrossEntropyLoss()

        self.label_lookup = {label: i for i, label in enumerate(self.labels)}
        self.batch_size = params['batch_size']
        self.use_gpu = params['use_gpu']

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'loss_state_dict': self.loss_function.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.params['paths']['save_path'], 'astnn_checkpoint.tar'))

    def load(self):
        checkpoint = torch.load(os.path.join(self.params['paths']['load_path'], 'astnn_checkpoint.tar'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.params['use_gpu']:
            self.model.cuda()
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adamax(parameters)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_function.load_state_dict(checkpoint['loss_state_dict'])

    def train(self):
        train_loss_, val_loss_, train_acc_, val_acc_ = [], [], [], []
        best_acc = 0.0
        print('Start training...')
        best_model = self.model
        total_epochs = self.params['epochs']
        for epoch in range(total_epochs):
            start_time = time.time()

            total_loss, total_acc, total = self._train_epoch(self.train_data, True)
            train_loss_.append(total_loss / total)
            train_acc_.append(total_acc / total)

            total_loss, total_acc, total = self._train_epoch(self.val_data, False)
            val_loss_.append(total_loss / total)
            val_acc_.append(total_acc / total)

            end_time = time.time()
            if total_acc/total > best_acc:
                best_model = self.model
                best_acc = total_acc
            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
                  % (epoch + 1, total_epochs, train_loss_[epoch], val_loss_[epoch],
                     train_acc_[epoch], val_acc_[epoch], end_time - start_time))

        self.model = best_model
        total_loss, total_acc, total = self._train_epoch(self.test_data, False)
        print("Testing results(Acc):", total_acc / total)

        print('Saving model')
        self.save()

    def evaluate(self):
        #TODO fix data reading in model class
        data = []
        total_loss, total_acc, total = self._train_epoch(data, False)
        print("Evaluating results, Accuracy:", total_acc / total)
        print("Evaluating results, Loss:", total_loss / total)

    def predict(self):
        #TODO fix data reading in model class and everything
        data_source = []
        total = 0
        inputs = []
        i = 0
        for code, _ in data_source:
            inputs.append(code)
            i += 1
            if i < self.batch_size:
                continue

            self.model.batch_size = len(inputs)
            self.model.hidden = self.model.init_hidden()
            output = self.model(inputs)
            _, predicted = torch.max(output.data, 1)

            total += len(inputs)
            print(predicted)

            inputs = []
            i = 0
        if i > 0:
            self.model.batch_size = len(inputs)
            self.model.hidden = self.model.init_hidden()
            output = self.model(inputs)
            _, predicted = torch.max(output.data, 1)

            total += len(inputs)
            print(predicted)

        print("Predicted labels:", total)

    def _step(self, inputs: List, labels: torch.LongTensor, is_train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.LongTensor(labels)
        if self.use_gpu:
            labels = labels.cuda()
        if is_train:
            self.model.zero_grad()
        self.model.batch_size = len(labels)
        self.model.hidden = self.model.init_hidden()
        output = self.model(inputs)
        loss = self.loss_function(output, Variable(labels))
        if is_train:
            loss.backward()
            self.optimizer.step()
        _, predicted = torch.max(output.data, 1)
        return predicted, loss

    def _train_epoch(self, data_source: CodeDataset, is_train: bool) -> Tuple[float, float, float]:
        total_acc, total_loss, total = 0., 0., 0.
        i = 0
        inputs, labels = [], []
        for code, label in data_source:
            inputs.append(code)
            labels.append(self.label_lookup[label])
            i += 1
            if i < self.batch_size:
                continue
            labels = torch.LongTensor(labels)
            predicted, loss = self._step(inputs, labels, is_train)
            total_acc += (predicted == labels).sum()
            total += len(labels)
            total_loss += loss.item() * len(inputs)

            inputs, labels = [], []
            i = 0
        if i > 0:
            labels = torch.LongTensor(labels)
            predicted, loss = self._step(inputs, labels, is_train)
            total_acc += (predicted == labels).sum()
            total += len(labels)
            total_loss += loss.item() * len(inputs)

        return total_loss, total_acc, total
