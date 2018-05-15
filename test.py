import torch
from torch.autograd import Variable
import time


def model_test(model, test_data):
    # model.cuda()
    start_time = time.time()
    time.time()
    for (images, labels) in test_data:
        # images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    end_time = time.time()
    execution_time = end_time - start_time
    print "execution time on GPU is: {}".format(execution_time)
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Testing is Done!')




