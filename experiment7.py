import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
job_id = os.environ["SLURM_JOB_ID"]

import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image as im
import math
from os import listdir as ld
import random

def im_square_crop(img_file):
    img = im.open(img_file)
    oldsize = img.size
    newsize = tuple(min(oldsize) for i in range(2))
    coords = list(math.floor((oldsize[i]-newsize[i])/2) for i in range(2))
    for i in range(2):
        coords.append(coords[i]+newsize[0])
    for i in range(len(coords)):
        coords[i] = round(coords[i])
    img = img.crop(tuple(coords))
    return img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(147456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 147456)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# def imshow(img, ax):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if npimg.shape == (512, 512):
#         npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis = 2)
#     ax.imshow(npimg)


class SunDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, data_file, train = False, test = False):
        super(SunDataset).__init__()

        def makeList(input, slice, split):
            output = list(i for i in input)
            for i, x in enumerate(output):
                output[i] = x[slice[0]:slice[1]].split(split)
                bl = []
                for j, y in enumerate(output[i]):
                    try:
                        output[i][j] = float(y)
                    except ValueError:
                        bl.append(y)
                for k in bl:
                    output[i].remove(k)
                    bl.remove(k)                  
            return output
        
        def cutset(dataset, sampling, inv = False):
            newset = []
            if not inv:
                for i in range(0, len(dataset), sampling):
                    newset.append(dataset[i])
            elif inv:
                x = list(range(len(dataset)))
                for i in range(0, len(dataset), sampling):
                    x.remove(i)
                for i in x:
                    newset.append(dataset[i])
            return newset

        self.input_dir = input_dir
        self.images = []
        for i in ld(self.input_dir):
            for j in ld("{}/{}".format(self.input_dir, i)):
                x = tuple("{}/{}/{}/{}".format(self.input_dir, i, j, k) for k in ld("{}/{}/{}".format(self.input_dir, i, j)))
                if len(x) == 9:
                    self.images.append(x)
        f = open(data_file, "r")
        m = open("./data_sun/missing_days.txt", "r")
        ml = m.readlines()
        ml = makeList(ml, [6, 16], "/")
        new_ml = []
        for i in ml:
            if i not in new_ml:  
                new_ml.append(i) 

        ml = new_ml 
        # for i in ml:
        #     if ml.count(i) != 1:
        #         print(i)
        #         print(ml.count(i))

        # check = makeList()

        self.data = f.readlines()
        checker = makeList(self.data, [0, 10], " ")
                
        for x in ml:
            try:
                ind = checker.index(x)
                del checker[ind]
                del self.data[ind]
            except ValueError:
                pass
            
        # for i in checker:
        #     for j, x in enumerate(i):
        #         if len(str(int(x))) == 1:
        #             i[j] = "0" + str(int(x))
        #         else:
        #             i[j] = str(int(x))
        #     path = "./data_sun/images/{}/{}".format(i[0], i[1] + i[2])
        #     if not os.path.isdir(path):
        #         print(path)

        for i, data in enumerate(self.data):
            self.data[i] = float(data[21:24])

        if train:
            self.images = cutset(self.images, 5, inv = True)
            self.data = cutset(self.data, 5, inv = True)
        elif test:
            self.images = cutset(self.images, 5)
            self.data = cutset(self.data, 5)
        


    def __getitem__(self, idx):
        a = tuple(0.5 for i in range(len(self.images[idx])))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(a, a)])
        images = np.zeros((len(self.images[idx]), 412, 412, 3))
        for i, x in enumerate(self.images[idx]):
            y = np.array(im.open(x))
            if y.shape == (412, 412):
                images[i] = np.broadcast_to(y.reshape(412, 412, 1), (412, 412, 3))
            else:
                images[i] = np.array(im.open(x))
        images = np.average(images, axis = 3)
        images = np.array(images, dtype = "uint8")
        images = np.transpose(images, (1, 2, 0))
        images = transform(images)
        val = torch.Tensor([np.double(self.data[idx])])
        return (images, val)

    def __len__(self):
        return len(self.data)

trainset = SunDataset("./data_sun/images_crop", "./data_sun/sunspots/dailySunspotNumber.txt", train = True)
testset = SunDataset("./data_sun/images_crop", "./data_sun/sunspots/dailySunspotNumber.txt", test = True)
dataset = SunDataset("./data_sun/images_crop", "./data_sun/sunspots/dailySunspotNumber.txt")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

# k = random.randint(0, len(dataset.images))

# figs, axs = plt.subplots(3, 3)
# for i in range(9):
#     imshow(dataset[k][0][i], axs[int(i/3)][i%3])
#     axs[int(i/3)][i%3].set_yticks([])
#     axs[int(i/3)][i%3].set_xticks([])
#     axs[int(i/3)][i%3].set_title("{} $\AA$".format(dataset.images[k][i][-8:-4]))
# plt.suptitle("Images of the Sun on {}".format(k))
# print(dataset[k][1])
# plt.show()


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda")
# print(torch.cuda.is_available())

net = Net()

net.to(device = device)

loader = torch.load("./solar_net_1957164/solar_epoch_ 1.pth")
state_dict = net.state_dict()
for i in loader.keys():
    state_dict[i] = loader[i]
net.load_state_dict(state_dict)

lr = 0.00001
mom = 0.01
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom)

# fig, ax  = plt.subplots()
# line, = ax.plot(0, 0)
# ax.set_autoscale_on(True)

# print(lr)
# print(mom)
for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        rloss = running_loss/(i + 1)

#        ax.scatter(epoch*len(trainloader) + i + 1, rloss, c = "blue")
#        plt.pause(0.05)

        # print(rloss)
        # print every 2000 mini-batches
##        if i % len(trainloader) == len(trainloader)-1:    
##            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / len(trainloader)))
##            running_loss = 0.0

        # x = np.concatenate((line.get_xdata(),[i]))
        # y = np.concatenate((line.get_ydata(),[rloss]))
        # line.set_data(x,y)
        # ax.autoscale_view())
    total = 0.0
    with torch.no_grad():
        for datax in testloader:
            images, targets = datax[0].to(device), datax[1].to(device)
            outputs = net(images)
            predicted = outputs.data
            measure = criterion(predicted, targets)
            total += measure.item()
    print("[epoch: %d] loss: %.5f, testset loss: %.5f" % (epoch + 1, running_loss / (len(trainloader)*np.var(np.array(trainset.data))),
                                                          total / (len(testloader) * np.var(np.array(testset.data)))))
    if not os.path.isdir("./solar_net_{}".format(job_id)):
        os.mkdir("./solar_net_%s" % (job_id))
    PATH = "./solar_net_%s/solar_epoch_%2d.pth" % (job_id, epoch + 1)
    torch.save(net.state_dict(), PATH)

# PATH = "./solar_net_1957151/solar_epoch_9.pth"
# net.load_state_dict(torch.load(PATH))
# print(net.parameters)

# plt.show()

# print('Finished Training')

# net = Net()
# PATH = './image_classif_net.pth'
# net.load_state_dict(torch.load(PATH))

# img = im_square_crop('cat_test.jpg')
# plt.imshow(img)
# plt.show()
# print(img.size)

# correct = 0
# total = 0
# loss = nn.MSELoss()
# sdir = "./solar_net_1957151" # % (job_id)
# for i, x in enumerate(ld(sdir)):
#     net.load_state_dict(torch.load("{}/{}".format(sdir, x)))
#     with torch.no_grad():
#         for data in testloader:
#             images, targets = data[0].to(device), data[1].to(device)
#             outputs = net(images)
#             predicted = outputs.data
#             measure = loss(predicted, targets)
#             total += measure.item()
#     print("epoch: %d loss on testset: %.3f" % (i+1, total / (len(testloader) * np.var(np.array(testset.data)))))
        


#print('Accuracy of the network on the 648 test images: %d %%' % (100 * correct / total))
