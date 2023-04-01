import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = (x_grad * x_grad + y_grad * y_grad) ** 0.5
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """
    ZS = ((grad_dir >= np.pi / 8 * 1) & (grad_dir < np.pi / 8 * 3) | (grad_dir >= -np.pi / 8 * 7) & (grad_dir < -np.pi / 8 * 5))
    S  = ((grad_dir >= np.pi / 8 * 3) & (grad_dir < np.pi / 8 * 5) | (grad_dir >= -np.pi / 8 * 5) & (grad_dir < -np.pi / 8 * 3))
    YS = ((grad_dir >= np.pi / 8 * 5) & (grad_dir < np.pi / 8 * 7) | (grad_dir >= -np.pi / 8 * 3) & (grad_dir < -np.pi / 8 * 1))
    O  = ((grad_dir >= np.pi / 8 * 7) | (grad_dir <-np.pi / 8 * 7) | (grad_dir >= -np.pi / 8 * 1) & (grad_dir <  np.pi / 8 * 1))
    
    n, m = grad_mag.shape
    
    T = np.zeros(grad_mag.shape)
    T[1:,1:] = grad_mag[:n-1,:m-1]
    ZS = ZS & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[:n-1,:m-1] = grad_mag[1:,1:]
    ZS = ZS & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[1:,:] = grad_mag[:n-1,:]
    S = S & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[:n-1,:] = grad_mag[1:,:]
    S = S & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[1:,:m-1] = grad_mag[:n-1,1:]
    YS = YS & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[:n-1,1:] = grad_mag[1:,:m-1]
    YS = YS & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[:,1:] = grad_mag[:,:m-1]
    O = O & (grad_mag >= T)

    T = np.zeros(grad_mag.shape)
    T[:,:m-1] = grad_mag[:,1:]
    O = O & (grad_mag >= T)

    NMS_output = grad_mag * (ZS | S | YS | O)
    return NMS_output


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.7
    high_ratio = 1.5
    
    n, m = img.shape
    avr = np.sum(img) / np.sum(img > 0)
    output = (img > avr * high_ratio)

    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    Q = [0]
    for i in range(n):
        for j in range(m):
            if output[i][j]:
                Q.append((i,j))
    
    he = 0
    ta = len(Q) - 1
    while he != ta:
        he += 1
        x = Q[he]
        for d in range(8):
            y = (x[0] + dx[d], x[1] + dy[d])
            if y[0] < 0 or y[0] > n - 1 or y[1] < 0 or y[1] > m - 1:
                continue
            if output[y[0]][y[1]]:
                continue
            if img[y[0]][y[1]] <= avr * low_ratio:
                continue
            output[y[0]][y[1]] = True
            ta += 1
            Q.append(y)

    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
