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
np.random.seed(7)

# training params
batch_size = 128
epochs = 500

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
            pokemon_image = cv2.resize(pokemon_image, (64, 64))
            pokemon_image = np.moveaxis(pokemon_image, 2, 0)
            X.append(pokemon_image)

# data loader for processing in batches
data_loader = DataLoader(X, batch_size=batch_size)

# covert output vectors to images if flag is true, else input images to vectors
def images_to_vectors(data, reverse=False):
    if reverse:
        return data.view(data.size(0), 3, 64, 64)
    else:
        return data.view(data.size(0), 12288)

# Generator model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        d = 128
        
        self.model = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(1000, d*8, 4, 1, 0),
                torch.nn.BatchNorm2d(d*8),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
                torch.nn.BatchNorm2d(d*4),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
                torch.nn.BatchNorm2d(d*2),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(d*2, d, 4, 2, 1),
                torch.nn.BatchNorm2d(d),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(d, 3, 4, 2, 1),
                torch.nn.Dropout2d(.5),
                torch.nn.Tanh()
        )


    def forward(self, x):
        img = self.model(x)
        return img

    def noise(self, s):
       #x = Variable(torch.randn(s, 100))
       x = torch.randn((5*5, 1000)).view(-1, 1000, 1, 1)
       return x


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        d = 128

        self.model = torch.nn.Sequential(
                torch.nn.Conv2d(3, d, 4, 2, 1),
                torch.nn.LeakyReLU(.2),
                torch.nn.Conv2d(d, d*2, 4, 2, 1),
                torch.nn.BatchNorm2d(d*2),
                torch.nn.LeakyReLU(.2),
                torch.nn.Conv2d(d*2, d*4, 4, 2, 1),
                torch.nn.BatchNorm2d(d*4),
                torch.nn.LeakyReLU(.2),
                torch.nn.Conv2d(d*4, d*8, 4, 2, 1),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm2d(d*8),
                torch.nn.Conv2d(d*8, 1, 4, 1, 0),
                torch.nn.Dropout2d(.5),
                torch.nn.Sigmoid()
        )


    def forward(self, img):
        output = self.model(img)
        return output


# discriminator training
def train_discriminator(discriminator, optimizer, real_data, fake_data):
    Nr = real_data.size(0)
    Nf = fake_data.size(0)
    optimizer.zero_grad()

    # train on 500
    # get prediction

    # figure out how to shape the input data to the discriminator
    pred_real = discriminator(real_data)

    # calculate loss
    error_real = loss_fx(pred_real, Variable(torch.ones(Nr, 1)))
    
    # calculate gradients
    error_real.backward()

    # train on fake
    # get prediction
    pred_fake = discriminator(fake_data)

    # calculate loss
    error_fake = loss_fx(pred_fake, Variable(torch.zeros(Nf, 0)))

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


def checkpoint():
    # save image sample and generator weights

    paths = [r'./dcgan_checkpoints', 'images', 'weights']

    if not os.path.exists(paths[0]):
       os.mkdir(paths[0])
    if not os.path.exists('{0}/{1}'.format(paths[0], paths[1])):
       os.mkdir('{0}/{1}'.format(paths[0], paths[1]))
    if not os.path.exists('{0}/{1}'.format(paths[0], paths[2])):
       os.mkdir('{0}/{1}'.format(paths[0], paths[2]))

    cv2.imwrite(r'./dcgan_checkpoints/images/epoch-{0}_batch-{1}_sample.jpg'.format(epoch, n_batch), test_img)
    torch.save(generator, r'./dcgan_checkpoints/weights/generator_epoch-{0}_batch-{1}.pth'.format(epoch, n_batch))


if __name__ == '__main__':
    # Instance of generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    
    # optimizers
    # .0005 glr .01 dlr seems to work well
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.002)

    # generator replay buffer
    replay_buffer = deque(maxlen=26)
    n_replay_train = 0
    
    # training loop
    for epoch in range(epochs):
         for n_batch, batch in enumerate(data_loader, 0):
             N = batch.size(0)
    
             real_images = Variable(batch).float()
             fake_images = generator(generator.noise(N)).detach()

             # add to replay buffer
             replay_buffer.append((real_images, fake_images))
              
             # additional training for discriminator on replay buffer every couple epochs
             if n_batch % 6 == 0 and len(replay_buffer) > 0:
                 n_replay_train += 1
                 buffer_selection = replay_buffer[np.random.choice(len(replay_buffer))]
                 print('Training on replay buffer. Total replay trains: {}'.format(n_replay_train))
                 _, _, _, = train_discriminator(
                         discriminator,
                         d_optimizer,
                         buffer_selection[0],
                         buffer_selection[1]
                 )

    
             # get error and train discriminator
             d_error, d_pred_real, d_pred_fake = train_discriminator(
                     discriminator,
                     d_optimizer,
                     real_images,
                     fake_images
             )
    
             # generate noise
             fake_data = generator.noise(N)
    
             # get error and train generator
             g_error = train_generator(generator, g_optimizer, fake_data)
    
             # convert generator output to image and preprocess to show
             test_img = np.array(images_to_vectors(generator(fake_data), reverse=True).detach())
             test_img = test_img[0, :, :, :]
             test_img = np.moveaxis(test_img, 0, 2)
    
             print('EPOCH: {0}, BATCH: {1} ====> D_ERROR: {2:.6f}, G_ERROR: {3:.6f}'.format(epoch, n_batch, d_error, g_error))

             # checkpoints images and weights
             checkpoint()
    
             # show example of generated image
             cv2.imshow('GENERATED', test_img)
             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break

    
    cv2.destroyAllWindows()
