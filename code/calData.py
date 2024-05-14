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
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')

    N = 10000

    if dataName == "iSUN": N = 8925
    print("Processing in-distribution images")

########################################In-distribution###########################################
    for j, data in enumerate(testloader10):
        
        if j<1000: continue
        images, _ = data

        # 이미지를 CUDA 장치로 보내고 Variable로 변환, requires_grad = True로 설정하여 그라디언트 계산 가능하게 함
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        nnOutputs = outputs.data.cpu()

        # tensor -> array
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)

        '''
            equ (1) => T = 1
        '''
        # ipdb.set_trace()
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

        # 계산된 확률 중 최대값을 파일에 기록
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        '''
            image perturbations를 위한 계산 
        '''
        # Using temperature scaling
        outputs = outputs / temper

        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))

        # 모델 학습에 사용되지 않고, 그라디언트를 얻기 위한 목적
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 그라디언트(inputs.grad.data)가 0보다 크거나 같으면 True, 작으면 False를 반환
        gradient =  torch.ge(inputs.grad.data, 0)
        # True/False 값을 1과 -1로 변환
        gradient = (gradient.float() - 0.5) * 2
        # 그라디언트 값을 실제 이미지의 픽셀 값과 같은 스케일로 조정하여, 교란을 더 자연스럽게 만듦
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)

        '''
            small perturbations
            : 계산된 그라디언트를 기반으로 교란을 추가하는 작업을 수행
            : 목적은 모델이 이런 작은 변화에도 불구하고 일관된 예측을 할 수 있는지를 테스트하기 위함
        '''

        # 원본 이미지(inputs.data)에 그라디언트 방향으로 노이즈를 추가하여 교란된 이미지를 생성
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))

        '''
            temperature
        '''

        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # 매 100번째 이미지 처리 시마다 처리 상태와 시간 출력
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        # visualization
        if j == 1000:  
            save_path = "/home/cal-05/hj/odin/results"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            visualize_difference(inputs.data[0].cpu(), tempInputs[0].cpu(), save_path, "ID.png")
        
        if j == N - 1: break


    t0 = time.time()
    print("Processing out-of-distribution images")

###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
    
        if j<1000: continue
        images, _ = data

        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)

        nnOutputs = outputs.data.cpu()

        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        
        # 소프트맥스를 통해 확률 계산
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temper
  
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

        '''
            Our
        '''

        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))

        # temperature
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
        if j == 1000: 
            save_path = "/home/cal-05/hj/odin/results"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            visualize_difference(inputs.data[0].cpu(), tempInputs[0].cpu(), save_path, "OOD.png")

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











