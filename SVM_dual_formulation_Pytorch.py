import numpy as np
import torch
import matplotlib.pyplot as plt


''' Start SVM helpers '''
def svm_contour(pred_fxn, xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33):
    '''
    Produces a contour plot for the prediction function.

    Arguments:
        pred_fxn: Prediction function that takes an n x d tensor of test examples
        and returns your SVM's predictions.
        xmin: Minimum x-value to plot.
        xmax: Maximum x-value to plot.
        ymin: Minimum y-value to plot.
        ymax: Maximum y-value to plot.
        ngrid: Number of points to be plotted between max and min (granularity).
    '''
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)),
            dim = 2).view(-1, 2)
        zz = pred_fxn(x_test)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                         cmap = 'coolwarm')
        plt.clabel(cs)
        plt.show()

def poly_implementation(x, y, degree):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (1 + (x * y).sum()).pow(degree)

def poly(degree):
    return lambda x, y: poly_implementation(x, y, degree)

def rbf_implementation(x, y, sigma):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (-(x - y).norm().pow(2) / 2 / sigma / sigma).exp()

def rbf(sigma):
    return lambda x, y: rbf_implementation(x, y, sigma)

def xor_data():
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y

''' End SVM Helpers '''

''' Start Logistic Helpers '''
def load_logistic_data():
    torch.manual_seed(0) # reset seed
    return linear_problem(torch.tensor([-1., 2.]), margin=1.5, size=200)

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
''' End Logistic Helpers '''

#Actual Model starts from here:

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computing an SVM model,  given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    alpha = torch.zeros_like(y_train, dtype=torch.float32, requires_grad=True)
    for i in range(num_iters):
        f = 0
        for j in range(x_train.shape[0]):
            for k in range(x_train.shape[0]):
                f = f + alpha[j]*alpha[k]*y_train[j]*y_train[k]*kernel(x_train[k],x_train[j])
        loss = 0.5*f - alpha.sum()
        alpha.retain_grad()
        loss.backward()
        with torch.no_grad():
            #alpha -= lr * (alpha.grad + c * (alpha - y_train * alpha.clamp(max=0)))
            alpha.sub_(lr*alpha.grad)
            alpha = alpha.clamp_( min = 0.0, max = c)
        alpha.grad.zero_()
    return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    pred = torch.zeros(x_test.shape[0])
    
    for i in range(x_test.shape[0]):
        value = 0
        for j in range(x_train.shape[0]):
            value = value + alpha[j]*y_train[j]*kernel(x_train[j],x_test[i])
        pred[i] = value 
    return (pred)
 

def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    lrate=torch.tensor(lrate)
    x = torch.cat((torch.ones((X.shape[0],1)),X), dim = 1)
    w = torch.zeros((X.shape[1]+1), requires_grad = True).reshape(X.shape[1]+1,1)
    for i in range(num_iter):
        y_hat = torch.matmul(x,w)
        loss = torch.sum(torch.log(torch.exp(-(y_hat*Y))+1))*(1/X.shape[0])
        w.retain_grad()
        loss.backward()
        with torch.no_grad():
            w-=(lrate*w.grad)
        w.grad.zero_()
    return w.detach()

