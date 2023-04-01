import numpy as np
from utils import draw_save_plane_with_points

def SVD(X):
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)

    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]

    # Extract a, b, c, d coefficients.
    x0, y0, z0 = C
    a, b, c = N
    d = -(a * x0 + b * y0 + c * z0)

    return a, b, c, d


if __name__ == "__main__":


    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")
    

    #RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0    

    sample_time = int(np.ceil(-np.log(1000) / np.log(1 - ((10 / 13) ** 3))))#more than 99.9% probability at least one hypothesis does not contain any outliers 
    distance_threshold = 0.05

    # sample points group
    points_group = np.random.randint(noise_points.shape[0], size=(sample_time, 3))



    # estimate the plane with sampled points group
    p2 = noise_points[np.broadcast_to(points_group[:,2:3], (sample_time, 3)) , np.broadcast_to(np.arange(3), (sample_time, 3))]
    p1 = noise_points[np.broadcast_to(points_group[:,1:2], (sample_time, 3)) , np.broadcast_to(np.arange(3), (sample_time, 3))]
    p0 = noise_points[np.broadcast_to(points_group[:,0:1], (sample_time, 3)) , np.broadcast_to(np.arange(3), (sample_time, 3))]
    v1 = p2 - p0
    v2 = p1 - p0
    cp = np.cross(v1, v2)
    a = cp[:,0:1]
    b = cp[:,1:2]
    c = cp[:,2:3]
    #d = - (cp[:,0:1] * p2[:,0:1] + cp[:,1:2] * p2[:,1:2] +cp[:,2:3] * p2[:,2:3])
    d = - np.sum(cp * p2, axis = 1, keepdims=True)


    #evaluate inliers (with point-to-plance distance < distance_threshold)
    dis = np.matmul(a, (noise_points[:,0:1]).T) + np.matmul(b, (noise_points[:,1:2]).T) + np.matmul(c, (noise_points[:,2:3]).T) + np.matmul(d, np.ones((1,noise_points.shape[0]),dtype=int))
    

    dis = np.abs(dis) / ((a*a+b*b+c*c) ** 0.5)
    inlier = (dis < distance_threshold)
    score = np.sum(inlier.astype(int), axis=1)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    id = np.argmax(score)
    I = np.arange(noise_points.shape[0])[inlier[id,:]]
    # X = np.zeros((I.shape[0],3))
    # X[:,0:1] = noise_points[I,0:1]
    # X[:,1:2] = noise_points[I,1:2]
    # X[:,2:3] = noise_points[I,2:3]
    X = noise_points[I,:]
    
    A,B,C,D = SVD(X)
    pf = [A,B,C,D]

    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

