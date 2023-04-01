import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.
    n,m = input_img.shape
    Ix = Sobel_filter_x(input_img)
    Iy = Sobel_filter_y(input_img)
    Ix2 = Ix * Ix
    Ixy = Ix * Iy
    Iy2 = Iy * Iy
    window = np.ones((window_size, window_size))
    Mx2 = convolve(Ix2, window)
    Mxy = convolve(Ixy, window)
    My2 = convolve(Iy2, window)
    theta = Mx2 * My2 - Mxy * Mxy - alpha * (Mx2 + My2) * (Mx2 + My2)
    row = np.broadcast_to(np.reshape(np.arange(n-window_size+1, dtype = int), (n-window_size+1, 1)), (n-window_size+1, m-window_size+1)) + window_size // 2
    col = np.broadcast_to(np.reshape(np.arange(m-window_size+1, dtype = int), (1, m-window_size+1)), (n-window_size+1, m-window_size+1)) + window_size // 2
    
    l = [row[theta > threshold], col[theta > threshold], theta[theta > threshold]]
    corner_list = list(zip(*l))

    return corner_list # the corners in corne_list: a tuple of (index of rows, index of cols, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 10

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
