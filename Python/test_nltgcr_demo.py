# -*- coding: utf-8 -*-

# Importing Libraries
import numpy as np
np.set_printoptions(precision=16)
import matplotlib.pyplot as plt
import numpy.linalg as nalg
import scipy.sparse.linalg




def nlTGCR(A, b, w, lb,epsf=1, iterations = 50, learning_rate = 1,
                     stopping_threshold = 1e-6):
     
    # Initializing weight, bias, learning rate and iterations

    iterations = iterations
    learning_rate = learning_rate

    d = len(b)
  
    P = np.zeros((d, lb))
    AP = np.zeros((d,lb))
    
    def FF(w):
         return learning_rate * (b-A@w)
    # w = np.random.random((100,))
    r = FF(w)

    rho = nalg.norm(r)
    ep = epsf * nalg.norm(w)/rho
    ep = 1e-15
    imagi = np.array([0+1.j])
    Ar = np.imag(FF(w-ep*r*imagi)/ep);
    # Ar = (FF( w-ep*r)-r)/ep
    t = nalg.norm(Ar)
    t = 1.0/t
    
    P[:,0] = t * r
    AP[:,0]=  t * Ar 
    # No restart
    restart = iterations+1 
    costs = []
    i2 = 1
    i = 1
    # Estimation of optimal parameters

    for it in range(1, iterations):
        
        # Calculationg the current cost


        alph = np.dot(np.transpose(AP),r)
        
        w = w + P@(alph)

        costs.append(nalg.norm(FF(w)))
        r = FF( w)
        rho = nalg.norm(r)
        # Ar = (FF( w-ep*r)-r)/ep
        Ar = np.imag(FF(w-ep*r*imagi)/ep);
        ep = epsf * nalg.norm(w)/rho
        
        p = r
        if i <= lb:
            k = 0
        else:
            k = i2
        while True:
            if k ==lb:
                k = 0
            k +=1
      
            tau = np.inner(Ar, AP[:,k-1])
            
            p = p - tau*(P[:,k-1])
            Ar = Ar -  tau*(AP[:,k-1])
     
            if k == i2:
                break
        t = nalg.norm(Ar)
        # if (it+1)% restart ==0:
        #     i2 =0
        #     i = 0
        #     P = np.zeros((d, lb))
        #     AP = np.zeros((d, lb))
        #     r = FF( w)
        #     rho = nalg.norm(r)
        #     Ar = (FF(w-ep*r)-r)/ep
        #     ep = epsf * nalg.norm(w)/rho
        #     t = nalg.norm(Ar)
        #     p = r
        if (i2) == lb:
            i2 = 0
        i2 = i2+1
        i = i+1
        t = 1.0/t
        AP[:,i2-1] = t*Ar
        P[:,i2-1] = t*p
    return w, costs



def conjugate_gradient(A, b, x=None, max_iter=50, reltol=1e-2, verbose=False):
    """
    Implements conjugate gradient method to solve Ax=b for a large matrix A that is not
    computed explicitly, but given by the linear function A. 
    """
    if verbose:
        print("Starting conjugate gradient...")
    if x is None:
        x=np.random.random((100,))
    # cg standard
    r=b-A(x)
    d=r
    rsnew=np.sum(r.conj()*r).real
    rs0=rsnew
    if verbose:
        print("initial residual: {}".format(rsnew))
    ii=0
    loss = []
    while ((ii<max_iter) and (rsnew>(reltol**2*rs0))):
        ii=ii+1
        Ad=A(d)
        alpha=rsnew/(np.sum(d.conj()*Ad))
        x=x+alpha*d
        if ii%50==0:
            #every now and then compute exact residual to mitigate
            # round-off errors
            r=b-A(x)
            d=r
        else:
            r=r-alpha*Ad
        rsold=rsnew
        rsnew=np.sum(r.conj()*r).real
        d=r+rsnew/rsold*d
        if verbose:
            print("{}, residual: {}".format(ii, rsnew))
        loss.append(nalg.norm(b-A(x)))
    return x, loss
 
res_gmres_rst = []
res_gmres = []
     
def gmres_rst_cl(r):
    res_gmres_rst.append(np.linalg.norm(r))
     
if __name__=="__main__":
    matrixSize = 100

    diag = np.random.rand(matrixSize).astype(np.float32)
    A_mat = np.random.random((100,100))
    A_mat = A_mat@A_mat.transpose() + 0.5*np.eye(100)
    A_mat = A_mat/np.linalg.norm(A_mat)
    b = np.ones_like(diag)
    xt = np.linalg.solve(A_mat, b)
    x00=np.random.rand(100,)    
    sol = scipy.sparse.linalg.gmres(A_mat, b, x0=x00, callback=gmres_rst_cl)
    lim = 100
    plt.semilogy(res_gmres_rst[:lim], marker='.',color='g', label='GMRES, full')
    
    b = np.ones((100,))
    x3, loss3 = nlTGCR(A_mat,b, x00,1)
    plt.semilogy(loss3,'r', label='NLTGCR. lb=1')
    plt.legend()
    x3, loss3 = nlTGCR(A_mat,b, x00,10)
    plt.semilogy(loss3,'b', label='NLTGCR. lb=10')
    plt.legend()