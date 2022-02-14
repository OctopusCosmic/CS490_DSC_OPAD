import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

# from __future__ import print_function
import torch  # pip3 install torch torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchsummary import summary  # pip3 install torch-summary

from torchvision import datasets, transforms


class TrafficNet(nn.Module):
    def __init__(self):
        super(TrafficNet, self).__init__()
        nclasses = 19  # 19 Big classes from GTSRB

        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250 * 2 * 2, 350)
        self.fc2 = nn.Linear(350, nclasses)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, 250 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Preprocess():
    def im2double(im):
        min_val = np.min(im.ravel())
        max_val = np.max(im.ravel())
        out = (im.astype('float') - min_val) / (max_val - min_val)
        return out

    def preprocess_image(img_filename, v, alpha):
        """
        processs the image with v matrix and noise
        input:
        img_filename: the path of an image in ppm format
        v matrix: 3 by 3 numpy array
        alpha: coefficient used for generating gaussian noise
        """
        # Comments on v matrix:
        # For some global color mixing matrix v, figure out what the input to the
        # projector should be, assuming zero illumination other than the projector.
        # Note : Not all scenes are possible for all v matrices. For some matrices
        # some scenes are not possible at all.

        img = cv2.imread(img_filename)  # numpy array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)

        # Captured Image - original #
        y = Preprocess.im2double(img)  # Convert to normalized floating point

        ss = y.shape  # (480,640,3)
        ylong = np.zeros((ss[0] * ss[1], 3))  # (307200,3)

        y1 = y[:, :, 0]  # (480,640)
        ylong[:, 0] = y1.flatten()
        y2 = y[:, :, 1]  # (480,640)
        ylong[:, 1] = y2.flatten()
        y3 = y[:, :, 2]  # (480,640)
        ylong[:, 2] = y3.flatten()

        xlong = np.transpose(np.matmul(np.linalg.pinv(v), np.transpose(ylong)))
        xlong[xlong > 1] = 1
        xlong[xlong < 0] = 0

        # Projector input - original #
        x = np.zeros(y.shape)
        x[:, :, 0] = xlong[:, 0].reshape(ss[0], ss[1])
        x[:, :, 1] = xlong[:, 1].reshape(ss[0], ss[1])
        x[:, :, 2] = xlong[:, 2].reshape(ss[0], ss[1])

        # Now you can get any perturbed image y = v(x+\delta x)

        xlong_new = xlong + alpha * np.random.rand(xlong.shape[0], xlong.shape[1])
        # Projector input - Attacked #
        x_new = np.zeros(x.shape)
        x_new[:, :, 0] = xlong_new[:, 0].reshape(ss[0], ss[1])
        x_new[:, :, 1] = xlong_new[:, 1].reshape(ss[0], ss[1])
        x_new[:, :, 2] = xlong_new[:, 2].reshape(ss[0], ss[1])

        ylong_new = np.transpose(np.matmul(v, np.transpose(xlong_new)))
        # Captured Image - Attacked #
        y_new = np.zeros(y.shape)
        y_new[:, :, 0] = ylong_new[:, 0].reshape(ss[0], ss[1])
        y_new[:, :, 1] = ylong_new[:, 1].reshape(ss[0], ss[1])
        y_new[:, :, 2] = ylong_new[:, 2].reshape(ss[0], ss[1])

        return y, x, y_new, x_new  # Captured original, Projector original, Captured attracked, Projector attacked

    def show_np_array_as_jpg(matrix, number):  # , original_filename, actual_label, v_number, n_number):
        # filename = "/home/zhan3447/CS490_DSC/jpg/{actual_label}/v{v_number}/n{n_number}/{original_filename}"
        filename = f"show{number}.jpg"
        # cv2.imwrite(filename, matrix) # does not work
        plt.imshow(matrix)
        plt.savefig(filename)


def get_predicted_label(img, device, model):  # numpy array get from the previous
    # Load the image
    image = np.zeros((3, 32, 32))  # (rgb, width, height) guess:)

    # add global.py code

    # temp = cv.imread('ISO_400.png')
    temp = img
    temp = cv2.resize(temp, (32, 32))  # resize the input image
    temp = temp[0:32, 0:32, :]

    temp = temp.astype('float64') / 255
    temp = temp[:, :, [2, 1, 0]]

    image[0, :, :] = temp[:, :, 0]
    image[1, :, :] = temp[:, :, 1]
    image[2, :, :] = temp[:, :, 2]

    # Convert the image to tensor
    data = torch.tensor(image)
    data = data.float()
    data = data.to(device)

    # Normalize the image
    data[0, :, :] = (data[0, :, :] - 0.485) / 0.229
    data[1, :, :] = (data[1, :, :] - 0.456) / 0.224
    data[2, :, :] = (data[2, :, :] - 0.406) / 0.225

    data = data.unsqueeze(0)

    data.requires_grad = False

    # Classify the image
    output = model(data)
    # print(torch.argmax(output))

    # Print output
    return torch.argmax(output).item()  # predicted label for the image


def main():
    df = pd.DataFrame()
    img_filename = "Test/0/10.ppm"
    # v = np.array([[1,1,1],
    #            [1,1,1],
    #            [1,1,1]])
    v = np.array([[1.0000, 0.0595, -0.1429],
                  [0.0588, 1.0000, -0.1324],
                  [-0.2277, -0.0297, 1.0000]])
    alpha = 0.05
    y, x, y_new, x_new = Preprocess.preprocess_image(img_filename, v, alpha)

    # show_np_array_as_jpg(y, 1)
    # show_np_array_as_jpg(x, 2)
    # show_np_array_as_jpg(y_new, 3)
    # show_np_array_as_jpg(x_new, 4)

    '''
    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
    '''

    epsilons = [0, .05, .1, .15, .2, .25, .3]
    use_cuda = True
    # Decide whether to use GPU or CPU

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Load the pretrained model
    model = TrafficNet()
    model = model.to(device)
    model.load_state_dict(torch.load('Traffic_sign_new/Checkpoints/epoch_999.pth'))

    model.eval()

    output_label_y = get_predicted_label(y, device, model)
    output_label_x = get_predicted_label(x, device, model)
    output_label_y_new = get_predicted_label(y_new, device, model)
    output_label_x_new = get_predicted_label(x_new, device, model)

    print(f'output_label_y: {output_label_y}')
    print(f'output_label_x: {output_label_x}')
    print(f'output_label_y_new: {output_label_y_new}')
    print(f'output_label_x_new: {output_label_x_new}')


main()