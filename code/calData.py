# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import os 
import ipdb

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def visualize_difference(original_img, perturbed_img, path, filename):
    
    # 이미지 정규화
    original_img = original_img * 0.5 + 0.5
    perturbed_img = perturbed_img * 0.5 + 0.5

    # 차이 계산
    difference = torch.abs(perturbed_img - original_img)

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(original_img.numpy(), (1, 2, 0)), interpolation='nearest')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.transpose(perturbed_img.numpy(), (1, 2, 0)), interpolation='nearest')
    plt.title('Perturbed Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.transpose(difference.numpy(), (1, 2, 0)), cmap='hot', interpolation='nearest')
    plt.title('Difference')
    plt.axis('off')

    # 결과 저장
    plt.savefig(os.path.join(path, filename))
    plt.close()

def testData(net1, criterion, CUDA_DEVICE, testloader10, testloader, nnName, dataName, noiseMagnitude1, temper):
    t0 = time.time()
    base_dir = f"./softmax_scores/{dataName}"

    # 경로가 존재하지 않으면 생성
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 파일 열기
    f1 = open(os.path.join(base_dir, "confidence_Base_In.txt"), 'w')
    f2 = open(os.path.join(base_dir, "confidence_Base_Out.txt"), 'w')
    g1 = open(os.path.join(base_dir, "confidence_Our_In.txt"), 'w')
    g2 = open(os.path.join(base_dir, "confidence_Our_Out.txt"), 'w')

    N = 10000

    if dataName == "iSUN": N = 8925
    print("Processing in-distribution images")

########################################In-distribution###########################################
    for j, data in enumerate(testloader10):
        
        if j<1000: continue
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()

        # tensor -> array
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)

        '''
            equ (1) => T = 1
        '''

        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
	
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)

        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        # visualization
        if j == 1000:  # 첫 변형된 이미지 저장
            save_path = "/home/cal-05/hj/odin/results"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            visualize_difference(inputs.data[0].cpu(), tempInputs[0].cpu(), save_path, "difference_image.png")
            break
        
        if j == N - 1: break


    t0 = time.time()
    print("Processing out-of-distribution images")

###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
    
        if j<1000: continue
        images, _ = data
    
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)


        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()

        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)

        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
  
  
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        # visualization
        if j == 1000:  # 첫 변형된 이미지 저장
            save_path = "/home/cal-05/hj/odin/results"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            visualize_difference(inputs.data[0].cpu(), tempInputs[0].cpu(), save_path, "OOD.png")
            break

        if j== N-1: break




def testGaussian(net1, criterion, CUDA_DEVICE, testloader10, testloader, nnName, dataName, noiseMagnitude1, temper):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
########################################In-Distribution###############################################
    N = 10000
    print("Processing in-distribution images")
    for j, data in enumerate(testloader10):
        
        if j<1000: continue
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

    
    
########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    for j, data in enumerate(testloader):
        if j<1000: continue
        
        images = torch.randn(1,3,32,32) + 0.5
        images = torch.clamp(images, 0, 1)
        images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
        images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
        images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
        
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break




def testUni(net1, criterion, CUDA_DEVICE, testloader10, testloader, nnName, dataName, noiseMagnitude1, temper):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
########################################In-Distribution###############################################
    N = 10000
    print("Processing in-distribution images")
    for j, data in enumerate(testloader10):
        if j<1000: continue
        
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4}  images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()



########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    for j, data in enumerate(testloader):
        if j<1000: continue
        
        images = torch.rand(1,3,32,32)
        images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
        images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
        images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
        
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break











