import torch
from torch.autograd import Variable
import time
import os
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

def train(init_model, train_data, epochs):
    init_model.train()
    # init_model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(init_model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(epochs):
        print "epoch : %d" % (epoch + 1)
        running_loss = 0
        for batch_index, (data, target) in enumerate(train_data):
            # data, target = data.cuda(), target.cuda()
            data, target = data, target

            inputs, targets = Variable(data), Variable(target)
            optimizer.zero_grad()
            outputs = init_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if batch_index % 100 == 0:
                print "[%d     , %5d] loss:%.4f" % (epoch + 1, batch_index, running_loss / 100)
                running_loss = 0

    torch.save(init_model, "./student.pth")
