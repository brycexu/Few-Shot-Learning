# Author : Bryce Xu
# Time : 2019/11/18
# Function: 主函数

from Parser import get_parser
from Dataset import dataloader
from Network import Network
from Loss import loss_fn
import torch
import numpy as np
from Logger import Logger
import os

logger = Logger('./logs')

parser = get_parser().parse_args()
print('--> Preparing Dataset:')
trainset = dataloader(parser, 'train')
valset = dataloader(parser, 'val')
testset = dataloader(parser, 'test')
print('--> Building Model:')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Network().to(device)
print('--> Initializing Optimizer and Scheduler')
optimizer = torch.optim.Adam(params=model.parameters(), lr=parser.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                            gamma=parser.lr_scheduler_gamma,
                                            step_size=parser.lr_scheduler_step)

def train_and_val(parser):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_model_path = os.path.join(parser.experiment_root, 'best_model.pth')
    best_state = model.state_dict()
    for epoch in range(parser.epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = []
        train_acc = []
        for batch_index, (inputs, targets) in enumerate(trainset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-parser.iterations:])
        avg_acc = np.mean(train_acc[-parser.iterations:])
        print('Traing Loss: {} | Accuracy: {}'.format(avg_loss, avg_acc))
        scheduler.step()
        for batch_index, (inputs, targets) in enumerate(valset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-parser.iterations:])
        avg_acc = 100. * np.mean(val_acc[-parser.iterations:])
        print('Validating Loss: {} | Accuracy: {}'.format(avg_loss, avg_acc))
        info = {'loss': avg_loss, 'accuracy': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
        if avg_acc > best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
    return best_state

def test(parser):
    avg_acc = []
    for epoch in range(10):
        for batch_index, (inputs, targets) in enumerate(testset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            _, acc = loss_fn(input=inputs, target=targets, n_support=parser.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = 100. * np.mean(avg_acc)
    print('Testing Accuracy: {}'.format(avg_acc))

if __name__ == '__main__':
    print('--> Begin Trainin and Validating')
    state = train_and_val(parser)
    print('--> Begin Testing')
    model.load_state_dict(state)
    test(parser)

