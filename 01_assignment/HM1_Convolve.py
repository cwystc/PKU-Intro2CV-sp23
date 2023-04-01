import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":
        n, m = img.shape
        padding_img = np.concatenate((np.zeros((n, padding_size)), img, np.zeros((n, padding_size))), axis = 1)
        padding_img = np.concatenate((np.zeros((padding_size, padding_size * 2 + m)), padding_img, np.zeros((padding_size, padding_size * 2 + m))), axis = 0)
        return padding_img
    elif type=="replicatePadding":
        n, m = img.shape
        S = np.concatenate((np.ones((padding_size, padding_size)) * img[0, 0], np.broadcast_to(img[:1, :], (padding_size ,m)), np.ones((padding_size, padding_size)) * img[0, m-1]), axis = 1)
        X = np.concatenate((np.ones((padding_size, padding_size)) * img[n-1, 0], np.broadcast_to(img[n-1: , :], (padding_size ,m)), np.ones((padding_size, padding_size)) * img[n-1, m-1]), axis = 1)
        Z = np.concatenate((np.broadcast_to(img[ : , :1], (n, padding_size)), img, np.broadcast_to(img[ : , m-1: ], (n, padding_size))), axis = 1)
        padding_img = np.concatenate((S, Z, X), axis = 0)
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, 1, "zeroPadding")

    #build the Toeplitz matrix and compute convolution
    T = np.zeros((36, 64))
    idx = np.reshape(np.reshape(np.arange(6) * 8, (6,1)) + (np.arange(6)), (36,))
    # idx = np.concatenate((idx, (idx + 8), (idx + 16), (idx + 24), (idx + 32), (idx + 40)), axis = None)
    
    sft = np.reshape(np.arange(3) * 8, (3,1)) + (np.arange(3))
    idx = np.reshape(idx, (36, 1, 1)) + sft
    T[np.broadcast_to(np.reshape(np.arange(36), (36, 1, 1)), (36, 3, 3)), idx] += kernel
    # for i in range(3):
    #     for j in range(3):
    #         T[np.arange(36), idx + j + i*8] += kernel[i,j]
    
    return np.reshape(np.matmul(T, np.reshape(padding_img, (64,1))), (6,6))

'''
def convolve_bf(img, kernel):
    n = img.shape[0]
    k = kernel.shape[0]
    res = np.zeros((n-k+1, n-k+1))
    for i in range(n-k+1):
        for j in range(n-k+1):
            for p in range(k):
                for q in range(k):
                    res[i, j] += kernel[p,q] * img[i+p,j+q]
    return res
'''
def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    n,m = img.shape
    k,l = kernel.shape
    H = np.broadcast_to(np.reshape(np.arange(n-k+1), (n-k+1,1,1,1)), (n-k+1,m-l+1,1,1)) + np.broadcast_to(np.reshape(np.arange(k), (k, 1)), (k, l))
    L = np.broadcast_to(np.reshape(np.arange(m-l+1), (1,m-l+1,1,1)), (n-k+1,m-l+1,1,1)) + np.broadcast_to(np.reshape(np.arange(l), (1, l)), (k, l))
    T = img[H, L]
    output = np.sum(np.sum(T * kernel, axis=3), axis=2)
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)
    

    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)
    '''
    result_bf = convolve_bf(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_bf.txt", result_bf)
    '''
    
    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    