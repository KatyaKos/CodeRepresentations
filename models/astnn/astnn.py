import os

import pandas as pd
import torch
import time
from models.astnn.model import BatchProgramClassifier
from torch.autograd import Variable

from models.model import CodeRepresentationModel


class ASTNN(CodeRepresentationModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = BatchProgramClassifier(self.EMBEDDING_DIM, config.HIDDEN_DIM, len(self.embedding),
                                            config.ENCODE_DIM, config.LABELS, config.BATCH_SIZE,
                                            config.USE_GPU, self.embedding)
        if config.LOAD_PATH:
            self.load()
        else:
            if config.USE_GPU:
                self.model.cuda()
            parameters = self.model.parameters()
            self.optimizer = torch.optim.Adamax(parameters)
            self.loss_function = torch.nn.CrossEntropyLoss()

        self.label_lookup = {label: i for i, label in enumerate(self.labels)}

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'loss_state_dict': self.loss_function.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.SAVE_PATH, 'astnn_checkpoint.tar'))

    def load(self):
        checkpoint = torch.load(os.path.join(self.config.LOAD_PATH, 'astnn_checkpoint.tar'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.config.USE_GPU:
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
        for epoch in range(self.config.EPOCHS):
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
                  % (epoch + 1, self.config.EPOCHS, train_loss_[epoch], val_loss_[epoch],
                     train_acc_[epoch], val_acc_[epoch], end_time - start_time))

        self.model = best_model
        total_loss, total_acc, total = self._train_epoch(self.test_data, False)
        print("Testing results(Acc):", total_acc / total)

        print('Saving model')
        self.save()

    def evaluate(self):
        data = pd.read_pickle(self.config.DATA_PATH)
        total_loss, total_acc, total = self._train_epoch(data, False)
        print("Evaluating results, Accuracy:", total_acc / total)
        print("Evaluating results, Loss:", total_loss / total)

    def predict(self):
        data = pd.read_pickle(self.config.DATA_PATH)
        total = 0
        i = 0
        while i < len(data):
            batch = self._get_batch(data, i, self.config.BATCH_SIZE, False)
            i += self.config.BATCH_SIZE
            inputs, _ = batch

            self.model.batch_size = len(inputs)
            self.model.hidden = self.model.init_hidden()
            output = self.model(inputs)

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total += len(inputs)
            print(predicted)

        print("Predicted labels:", total)

    def _onehot(self, i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]

    def _get_batch(self, dataset, idx, bs, with_labels=True):
        tmp = dataset.iloc[idx: idx + bs]
        data, labels = [], []
        for code, label in zip(tmp['code'], tmp['label']):
            data.append(code)
            if with_labels:
                labels.append(self.label_lookup[label])
        return data, torch.LongTensor(labels)

    def _train_epoch(self, data, is_train):
        total_acc, total_loss, total = 0., 0., 0.
        i = 0
        while i < len(data):
            batch = self._get_batch(data, i, self.config.BATCH_SIZE)
            i += self.config.BATCH_SIZE
            inputs, labels = batch
            if self.config.USE_GPU:
                inputs, labels = inputs, labels.cuda()

            if is_train:
                self.model.zero_grad()
            self.model.batch_size = len(labels)
            self.model.hidden = self.model.init_hidden()
            output = self.model(inputs)
            loss = self.loss_function(output, Variable(labels))
            if is_train:
                loss.backward()
                self.optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == labels).sum()
            total += len(labels)
            total_loss += loss.item() * len(inputs)

        return total_loss, total_acc.item(), total
