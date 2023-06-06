import torch
import numpy as np
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Data Creation
def load_reg_data():
    # load the regression synthetic data
    torch.manual_seed(0) # force seed so same data is generated every time

    X = torch.linspace(0, 4, 100).reshape(-1, 1)
    noise = torch.normal(0, .4, size=X.shape)
    w = 0.5
    b = 1.
    Y = w * X**2 + b + noise

    return X, Y

def load_xor_data():
    X = torch.tensor([[-1,1],[1,-1],[-1,-1],[1,1]]).float()
    Y = torch.prod(X,axis=1)

    return X, Y

def load_logistic_data():
    torch.manual_seed(0) # reset seed
    return linear_problem(torch.tensor([-1., 2.]), margin=1.5, size=200)

def contour_plot(xmin, xmax, ymin, ymax, pred_fxn, ngrid = 33):
    """
    make a contour plot
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param pred_fxn: prediction function that takes an (n x d) tensor as input
                     and returns an (n x 1) tensor of predictions as output
    @param ngrid: number of points to use in contour plot per axis
    """
    # Build grid
    xgrid = torch.linspace(xmin, xmax, ngrid)
    ygrid = torch.linspace(ymin, ymax, ngrid)
    (xx, yy) = torch.meshgrid(xgrid, ygrid)

    # Get predictions
    features = torch.dstack((xx, yy)).reshape(-1, 2)
    predictions = pred_fxn(features)

    # Arrange predictions into grid and plot
    zz = predictions.reshape(xx.shape)
    C = plt.contour(xx, yy, zz,
                    cmap = 'coolwarm')
    plt.clabel(C)
    plt.show()

    return plt.gcf()

def linear_problem(w, margin, size, bounds=[-5., 5.], trans=0.0):
    in_margin = lambda x: torch.abs(w.flatten().dot(x.flatten())) / torch.norm(w) \
                          < margin
    half_margin = lambda x: 0.6*margin < w.flatten().dot(x.flatten()) / torch.norm(w) < 0.65*margin
    X = []
    Y = []
    for i in range(size):
        x = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        while in_margin(x):
            x.uniform_(bounds[0], bounds[1]) + trans
        if w.flatten().dot(x.flatten()) + trans > 0:
            Y.append(torch.tensor(1.))
        else:
            Y.append(torch.tensor(-1.))
        X.append(x)
    for j in range(1):
    	x_out = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
    	while not half_margin(x_out):
            x_out = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
    	X.append(x_out)
    	Y.append(torch.tensor(-1.))
    X = torch.stack(X)
    Y = torch.stack(Y).reshape(-1, 1)

    return X, Y


#Actual Model Development

#First: Linear Regression using Gradient Descent approach
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform
    
    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    w = torch.zeros((X.shape[1]+1), requires_grad = True).reshape(X.shape[1]+1,1)
    for i in range(num_iter):
        y_hat = torch.matmul(x,w)
        loss = torch.square(torch.norm(y_hat-Y))/(2*X.shape[0])
        w.retain_grad()
        loss.backward()
        with torch.no_grad():
            w-=(lrate*w.grad)
        w.grad.zero_()
    return w

# Linear Regression using Least Square (Moore-Penrose pseudoinverse) Method; Instead of using Ordinary Least square method Moore-Penrose's method of using Pseudo-Inverse will give us the parameters of the linear regression model irrespective of "rank" of the matrix.
def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    M_plus = torch.pinverse(x)
    return torch.matmul(M_plus,Y)
    pass

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    import hw4_utils
    X,Y = (hw4_utils.load_reg_data())
    w =linear_normal(X,Y)
    plt.scatter(X,Y)
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    plt.plot(X,torch.matmul(x,w), color = "red")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    pass


def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    m,n = X.shape
    d = X.shape[1]
    for i in range(n):
        for j in range(i,n):
            X = torch.cat([X,(X[:,i]*X[:,j]).reshape(m,1)], dim =1)
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    new_shape = 1+d+(d*(d+1)/2)
    w = torch.zeros((int(new_shape),1), requires_grad = True)
    for i in range(num_iter):
        y_hat = torch.matmul(x,w)
        loss = torch.square(torch.norm(y_hat-Y))/(2*X.shape[0])
        w.retain_grad()
        loss.backward()
        with torch.no_grad():
            w-=(lrate*w.grad)
        w.grad.zero_()
    return w
    pass

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    m,n = X.shape
    d = X.shape[1]
    for i in range(n):
        for j in range(i,n):
            X = torch.cat([X,(X[:,i]*X[:,j]).reshape(m,1)], dim =1)
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    M_plus = torch.pinverse(x)
    return torch.matmul(M_plus,Y)
    pass

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    import hw4_utils
    X,Y = (hw4_utils.load_reg_data())
    w =poly_normal(X,Y)
    plt.scatter(X,Y)
    m,n = X.shape
    d = X.shape[1]
    for i in range(n):
        for j in range(i,n):
            X = torch.cat([X,(X[:,i]*X[:,j]).reshape(m,1)], dim =1)
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    plt.plot(X,torch.matmul(x,w), color = "red")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    pass


#Finding out the parameters of the model by expanding the feature space in more dimensions to find solution to XOR dataset which cannot be linearly seperable in two dimensional space.
def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    def pred_linear(X):
        x_linear = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
        #w_linear = linear_normal(X,Y)
        return torch.matmul(x_linear,w_linear)
    def pred_poly(X1):
        m,n = X1.shape
        d = X1.shape[1]
        for i in range(n):
            for j in range(i,n):
                X1 = torch.cat([X1,(X1[:,i]*X1[:,j]).reshape(m,1)], dim =1)
        x_poly = torch.cat((torch.ones((X1.shape[0],1)),X1), dim = 1)
        #w_poly = poly_normal(X,Y)
        return torch.matmul(x_poly,w_poly)
    import hw4_utils
    X,Y = (hw4_utils.load_xor_data())
    X1,Y = (hw4_utils.load_xor_data())
    x_linear = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    m,n = X.shape
    d = X1.shape[1]
    for i in range(n):
        for j in range(i,n):
            X1 = torch.cat([X1,(X1[:,i]*X1[:,j]).reshape(m,1)], dim =1)
    x_poly = torch.cat((torch.ones((X1.shape[0],1)),X1), dim = 1)
    w_linear = linear_normal(X,Y)
    w_poly = poly_normal(X,Y)
    return hw4_utils.contour_plot(min(X[:,0]),max(X[:,1]),min(Y),max(Y),pred_linear,ngrid = 33)
    

