from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

# for this file to work, you must have directory: 'uncropped', 'cropped', and 'A3Report'

act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson',    'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

# crop and resize a img with filename x(string), and cropping coordinate(string with ',' to seperate)
def crop(x, coordinate, filename):
    try:
        imRGB = imread(x)
    except IOError:
        return np.array([-1])
    if os.path.isfile('cropped/'+filename):
        return imread('cropped/'+filename)
    coordinate = coordinate.split(',')
    if (len(imRGB.shape) == 3):
        # turn image into grayscale
        img = np.dot(imRGB[...,:3], [0.30, 0.59, 0.11]).astype(np.uint8)
    else:
        img = imRGB
    # cropping: 'x1,y1,x2,y2' -> I[y1:y2, x1:x2]
    img = img[int(coordinate[1]): int(coordinate[3]), int(coordinate[0]): int(coordinate[2])]
    # resize to 32 x 32
    while img.shape[0] > 64:
        img = imresize(img, 0.5)
    img = imresize(img, (32, 32))
    return img
    
def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        v: eigen face
        s: eigen value
        #http://programmingcomputervision.com/
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X

def display_25_rand_images(im_matrix,im_shape):
    '''Display 25 components in V'''
    #gray()
    fig = figure()
    for i in range(25):
        num = random.randint(1, 799)
        im = array(im_matrix[num,:]).reshape(im_shape)
        subplot(5, 5, i+1)
        imshow(im)
        axis('off')
    savefig('A3Report/randim.jpg')  
    show()

def display_save_25_comps(V, im_shape):
    '''Display 25 components in V'''
    figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        gray()
        imshow(V[i,:].reshape(im_shape))
    savefig('A3Report/display_save_25_comps.jpg')  
    show() 

if __name__ == "__main__":
    testfile = urllib.URLopener()            
    
    # traImg = zeros((2560, 320)).astype(uint8) # for the image on report
    # valImg = zeros((256, 320)).astype(uint8) # for the image on report
    # testImg = zeros((256, 320)).astype(uint8) # for the image on report
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    actNum = 0
    for a in act:
        name = a.split()[1].lower()
        i = 0
        # plt.imshow(traImg)
        plt.pause(0.5)
        for line in open("faces_subset.txt"):
            if (a in line) and (i < 120):
                data = line.split()
                filename = name+str(i)+'.'+data[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                if not os.path.isfile("uncropped/"+filename):
                    img = timeout(testfile.retrieve, (data[4], "uncropped/"+filename), {}, 30)
                else:
                    img = ["uncropped/"+filename]
                if (not os.path.isfile("uncropped/"+filename)) or img is None:
                    continue
                    
                img = crop(img[0], data[-2], filename)
                if np.array_equal(img, [-1]):
                    continue
                # initializing
                if name == 'eckhart' and (i == 0 or i == 100 or i == 110):
                    if i == 0:
                        training = img.flatten()
                        # traImg[0:32, 0:32] = img
                    elif i == 100:
                        validation = img.flatten()
                        # valImg[0:32, 0:32] = img
                    else:
                        test = img.flatten()
                        # testImg[0:32, 0:32] = img
                else:
                    j = i // 10
                    rowSkip = actNum*320
                    if (i < 100):
                        # traImg[32*j + rowSkip : 32*(j+1) + rowSkip, 32*(i%10) : 32*((i%10)+1)] = img
                        training = np.vstack([training, img.flatten()])
                    elif (i >= 100 and i < 110):
                        # valImg[32*actNum : 32*(actNum+1), 32*(i - 100) : 32*(i - 99)] = img
                        validation = np.vstack([validation, img.flatten()])
                    elif (i >=110 and i < 120):
                        # testImg[32*actNum : 32*(actNum+1), 32*(i - 110) : 32*(i - 109)] = img
                        test = np.vstack([test, img.flatten()])
                # save img
                img = Image.fromarray(img)
                img.save('cropped/'+filename)
                
                print filename
                i += 1
        actNum += 1
    
#     # save data set image
#     x = Image.fromarray(traImg)
#     x.save('A3Report/trainingSet.jpg')
#     y = Image.fromarray(valImg)
#     y.save('A3Report/validationSet.jpg')
#     z = Image.fromarray(testImg)
#     z.save('A3Report/testSet.jpg')
#     
    result = pca(training)
    
    # save mean face and 25 eigenfaces
    mean_x = result[2].reshape((32, 32))
    mean_x = Image.fromarray(mean_x)
    mean_x.save('mean_face.jpg')
    display_25_rand_images(training, (32, 32))
    display_save_25_comps(result[0], (32, 32))
    mean_x = result[2]
    V = result[0]
    
    ## part 3: validation set
    print "Part 3 validation set resutl:"
    kList = [2, 5, 10, 20, 50, 80, 100, 150, 200]
    best = [-1, -1]
    for k in kList:
        correct = 0
        alpha_y = zeros((800, k))
        for y in range(training.shape[0]):
                alpha_y[y] = [np.dot(V[i,:], (training[y] - mean_x)) for i in range(k)]
        for x in range(validation.shape[0]):
            input = validation[x]
            alpha_x = np.array([np.dot(V[i,:], (input-mean_x)) for i in range(k)])
            dist = zeros(800)
            for d in range(800):
                dist[d] = np.linalg.norm(alpha_y[d] - alpha_x)
            idx = argmin(dist)
#             print x // 10, idx
            if (x // 10) == (idx // 100):
                correct += 1
        probCorrect = correct / 80.0
        if probCorrect > best[1]:
            best[1] = probCorrect
            best[0] = k
        print 'k = ' + str(k) + ': ' + str(probCorrect)
    
    ## test set
    correct = 0
    k = best[0]
    alpha_y = zeros((800, k))
    countFail = 0
    lefttop = 0
    for y in range(training.shape[0]):
            alpha_y[y] = [np.dot(V[i,:], (training[y] - mean_x)) for i in range(k)]
    for x in range(test.shape[0]):
        input = test[x]
        alpha_x = np.array([np.dot(V[i,:], (input-mean_x)) for i in range(k)])
        dist = zeros(800)
        for d in range(800):
            dist[d] = np.linalg.norm(alpha_y[d] - alpha_x)
        idx = argmin(dist)
#             print x // 10, idx
        if (x // 10) == (idx // 100):
            correct += 1
        elif countFail not in [2, 7, 19, 25, 33]:
            countFail += 1
        else:
            failCase[0:32, lefttop:lefttop+32] = training[idx].reshape(32, 32)
            lefttop += 32
            failCase[0:32, lefttop:lefttop+5] = 255
            lefttop += 5
            countFail += 1
    probCorrect = correct / 80.0
    fail = Image.fromarray(failCase)
    fail.save('A3Report/failCases.jpg')
    print 'Part 4 test set correct rate: ' + str(probCorrect)
    
    ## Part 4
    print "Part 3 validation set resutl:"
    # female = [6, 7, 3, 5, 4], therefore female: > 2, maile: <= 2
    kList = [2, 5, 10, 20, 50, 80, 100, 150, 200]
    best = [-1, -1]
    for k in kList:
        correct = 0
        alpha_y = zeros((800, k))
        for y in range(training.shape[0]):
                alpha_y[y] = [np.dot(V[i,:], (training[y] - mean_x)) for i in range(k)]
        for x in range(validation.shape[0]):
            input = validation[x]
            alpha_x = np.array([np.dot(V[i,:], (input-mean_x)) for i in range(k)])
            dist = zeros(800)
            for d in range(800):
                dist[d] = np.linalg.norm(alpha_y[d] - alpha_x)
            idx = argmin(dist)
            idx1 = x // 10
            idx2 = idx // 100
            if (idx1 <= 2 and idx2 <= 2) or (idx1 > 2 and idx2 > 2):
                correct += 1
        probCorrect = correct / 80.0
        if probCorrect > best[1]:
            best[1] = probCorrect
            best[0] = k
        print 'k = ' + str(k) + ': ' + str(probCorrect)
    
    ## test set
    correct = 0
    k = best[0]
    alpha_y = zeros((800, k))
    for y in range(training.shape[0]):
            alpha_y[y] = [np.dot(V[i,:], (training[y] - mean_x)) for i in range(k)]
    for x in range(test.shape[0]):
        input = test[x]
        alpha_x = np.array([np.dot(V[i,:], (input-mean_x)) for i in range(k)])
        dist = zeros(800)
        for d in range(800):
            dist[d] = np.linalg.norm(alpha_y[d] - alpha_x)
        idx = argmin(dist)
        idx1 = x // 10
        idx2 = idx // 100
        if (idx1 <= 2 and idx2 <= 2) or (idx1 > 2 and idx2 > 2):
            correct += 1
    probCorrect = correct / 80.0
    print 'Part 4 test set correct rate: ' + str(probCorrect)
                
    