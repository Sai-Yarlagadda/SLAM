'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """

    while(angle_rad>np.pi):
        angle_rad = angle_rad - 2* (np.pi)
    
    while(angle_rad < -np.pi):
        angle_rad = angle_rad + 2*(np.pi)
    
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    #Find the initial position of the robot from the init pose which is given
    x = init_pose[0][0]
    y = init_pose[1][0]
    theta = init_pose[2][0]
    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    #finding the landmarks
    for i in range(k):
        #getting landmark values
        beta = init_measure[2*i][0]
        r = init_measure[2*i + 1][0]
        lx = x + r * np.cos(theta + beta)
        ly = y + r * np.sin(theta + beta)
        landmark[i*2][0] = lx
        landmark[i*2 + 1][0] = ly
        
        #getting landmark covariances
        dlx_db = -r* np.sin(theta + beta)
        dlx_dr = np.cos(theta + beta)
        dly_db = r* np.cos(theta + beta)
        dly_dr = np.sin(theta + beta)
        H = np.array([[dlx_db, dlx_dr], [dly_db, dly_dr]]) #covariance matrix for a particular landmark
        landmark_cov[i*2:i*2+2, i*2:i*2+2] = H@ init_measure_cov @ H.T

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    x = X[0][0]
    y = X[1][0]
    theta = X[2][0]
    dt = control[0][0]
    alpha = control[1][0]
    cov2 = np.zeros((15,15))

    # getting the x at time t=t+1
    X_pre = np.zeros((3+ 2*k,1))
    X_pre[0][0] = x + dt* np.cos(theta)
    X_pre[1][0] = y + dt* np.sin(theta)
    X_pre[2][0] = theta + alpha
    X_pre[3:2*k+3,0] = X[3:2*k+3,0]

    #getting covariance of the above predicted X_pre
    Ft = np.eye(3+2*k)
    Ft[:3,:3] = [[1, 0, -dt* np.sin(theta)],[0, 1, dt* np.cos(theta)],[0, 0, 1]]
    Fu = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    #print(Ft.shape)
    #print(P.shape)
    #print(cov2.shape)

    cov2[0:3,0:3] = Fu@ control_cov @ Fu.T
    P_pre = (Ft @ P @ Ft.T) +  cov2 # assuming control uncertainity is linear

    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    
    updating the state matrix
    X = X_pre + K(measure - h(X_pre))
    K = P_pre. (H_T). (H.P_pre.(H_T) + Q)-^-1
    where X_pre = predicted shape vector
    K is Kalmaan Gain
    H is the measurement Jacobian for a specific landmark associated with the measurement
    R is the covariance matrix
    h(X_pre) is the predicted measurement based on current state estimate

    updating the covariance matrix
    P = (I - K.H).P_pre
    where I is an identity matrix
    K is Kalmaan Gain
    H is measurement Jacobian
    P_pre is the predicted covariance

    z = measure
    K = Kalman Gain
    '''
    H = np.zeros((2*k ,3+2*k))
    Q = np.zeros((2*k, 2*k))
    H_exp = np.zeros((2*k,1))
    x = X_pre[0][0]
    y = X_pre[1][0]
    theta = X_pre[2][0]  
    for i in range (k):
        lx = X_pre[3+2*i][0]
        ly = X_pre[3+2*i+1][0]
        m =( ((lx - x)**2)+ ((ly - y)**2))

        db_dxt = (ly-y)/m
        dr_dxt = (x-lx)/np.sqrt(m)
        db_dyt = (x-lx)/m
        dr_dyt = (y-ly)/np.sqrt(m)
        db_dtheta = -1
        dr_dtheta = 0
        db_dlx = (y-ly)/m
        dr_dlx = (lx-x)/np.sqrt(m)
        db_dly = (lx-x)/m
        dr_dly = (ly-y)/np.sqrt(m)

        H[2*i:2*i + 2, 0: 3] = np.block([[db_dxt, db_dyt, db_dtheta],[dr_dxt, dr_dyt, dr_dtheta]])
        H[2*i: 2*i + 2, 2*i+3: 2*i+5]  = np.block([[db_dlx, db_dly],[dr_dlx, dr_dly]])
        

        Q[2*i: 2*i+2, 2*i: 2*i+2] = measure_cov
        t = np.arctan2((ly-y),(lx-x))
        H_exp[2*i][0] = warp2pi(t - theta)
        H_exp[2*i+1][0] = np.sqrt(m)
    
    K = P_pre @ (H.T) @np.linalg.inv((H @ P_pre @ (H.T)) + Q)
    final_X_pre = X_pre + K@(measure - H_exp)
    final_P_pre = (np.eye(3 + 2*k) - K @ H)@(P_pre)

    return final_X_pre, final_P_pre


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    
    Euclidean_dist = np.zeros(k)
    Mahalanobis_dist = np.zeros(k)
    print(f'{Euclidean_dist}')
    print(f'{Mahalanobis_dist}')

    for i in range(k):
        #print euclidean distance
        Euclidean_dist[i] = np.sqrt((l_true[i*2] - X[3+i*2][0])**2 + (l_true[i*2+1] - X[3+i*2+1][0])**2)
        
        #print mahalanobis distance 
        x_u = np.array([l_true[i*2] - X[3+i*2][0], l_true[i*2+1] - X[3+i*2+1][0]])
        sigma = P[3+i*2: 5+i*2, 3+i*2: 5+i*2]
        Mahalanobis_dist[i] = np.sqrt(x_u@(sigma)@(x_u.T))
    print(f"The Euclidean Matrix is {Euclidean_dist}")
    print(f"The Mahalanobis distance is {Mahalanobis_dist}")
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25;
    sig_y = 0.1;
    sig_alpha = 0.1;
    sig_beta = 0.01;
    sig_r = 0.08;

    '''sig_x = 0.25;
    sig_y = 0.1;
    sig_alpha = 0.1;
    sig_beta = 0.01;
    sig_r = 0.8;'''


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
