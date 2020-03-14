import numpy as np
import torch


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=20, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        criterion = self.loss_func

        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')

        running_train_loss = 0.0

        for epoch in range(num_epochs):
            for i, data in enumerate(train_loader):
                if i == iter_per_epoch:
                    break
                
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optim.zero_grad()

                outputs = model(inputs)
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                optim.step()

                running_train_loss += train_loss
                if i % 5 == 0:
                    _, preds = torch.max(outputs, 1)
                    targets_mask = labels >= 0
                    acc = np.mean((preds == labels)[targets_mask].data.cpu().numpy())
                    print('train loss:', train_loss.item()) 
                    print('train acc:', acc)
                    print()


            inputs, labels = next(iter(val_loader))
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            targets_mask = labels >= 0
            acc = np.mean((preds == labels)[targets_mask].data.cpu().numpy())
            print('val loss:', val_loss.item())
            print('val acc - epoch', i, ':', acc)

        print('FINISH.')
