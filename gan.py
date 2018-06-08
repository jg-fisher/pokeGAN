""" Generative Adversarial Network.  """

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import os
import cv2
from collections import deque


# reproducibility
np.random.seed(3)


# training params
batch_size = 25
epochs = 5000

# loss function
loss_fx = torch.nn.BCELoss()

# processing images
X = deque()
for img in os.listdir('pokemon_images'):
    if img.endswith('.png'):
        pokemon_image = cv2.imread(r'./pokemon_images/{}'.format(img))
        if pokemon_image.shape != (96, 96, 3):
            pass
        else:
            X.append(pokemon_image)

# data loader for processing in batches
data_loader = DataLoader(X, batch_size=batch_size)

# covert output vectors to images if flag is true, else input images to vectors
def images_to_vectors(data, reverse=False):
    if reverse:
        return data.view(data.size(0), 3, 96, 96)
    else:
        return data.view(data.size(0), 27648)

# Generator model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        n_features = 1000
        n_out = 27648
        
        self.model = torch.nn.Sequential(
                torch.nn.Linear(n_features, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, n_out),
                torch.nn.Sigmoid()
        )


    def forward(self, x):
        img = self.model(x)
        return img

    def noise(self, s):
       x = Variable(torch.randn(s, 1000))
       return x


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 27648
        n_out = 1

        self.model = torch.nn.Sequential(
                torch.nn.Linear(n_features, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, n_out),
                torch.nn.Sigmoid()
        )


    def forward(self, img):
        output = self.model(img)
        return output


# discriminator training
def train_discriminator(discriminator, optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    # train on real
    # get prediction
    pred_real = discriminator(real_data)

    # calculate loss
    error_real = loss_fx(pred_real, Variable(torch.ones(N, 1)))
    
    # calculate gradients
    error_real.backward()

    # train on fake
    # get prediction
    pred_fake = discriminator(fake_data)

    # calculate loss
    error_fake = loss_fx(pred_fake, Variable(torch.ones(N, 0)))

    # calculate gradients
    error_fake.backward()

    # update weights
    optimizer.step()
    
    return error_real + error_fake, pred_real, pred_fake


# generator training
def train_generator(generator, optimizer, fake_data):
    N = fake_data.size(0)

    # zero gradients
    optimizer.zero_grad()

    # get prediction
    pred = discriminator(generator(fake_data))

    # get loss
    error = loss_fx(pred, Variable(torch.ones(N, 0)))

    # compute gradients
    error.backward()

    # update weights
    optimizer.step()

    return error


# Instance of generator and discriminator
generator = Generator()
discriminator = Discriminator()

# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00002)

# training loop
for epoch in range(epochs):
     for n_batch, batch in enumerate(data_loader, 0):
         N = batch.size(0)

         # Train Discriminator

         # show real image
         cv2.imshow('REAL', np.array(batch[0]))

         # REAL
         real_images = Variable(images_to_vectors(batch)).float()

         # FAKE
         fake_images = generator(generator.noise(N)).detach()

         # TRAIN
         d_error, d_pred_real, d_pred_fake = train_discriminator(
                 discriminator,
                 d_optimizer,
                 real_images,
                 fake_images
         )

         # Train Generator

         # generate noise
         fake_data = generator.noise(N)

         # get error based on discriminator
         g_error = train_generator(generator, g_optimizer, fake_data)

         # convert generator output to image and preprocess to show
         test_img = np.array(images_to_vectors(generator(fake_data), reverse=True).detach())
         test_img = test_img[0, :, :, :]
         test_img = np.moveaxis(test_img, 0, 2)
         print(test_img.shape)

         print('EPOCH: {0}, BATCH: {3}, D error: {1}, G error: {2}'.format(epoch, d_error, g_error, n_batch))

         # show example of generated image
         cv2.imshow('GENERATED', test_img)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break



cv2.destroyAllWindows()

# save weights
# torch.save('weights.pth')
