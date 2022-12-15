import numpy as np

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
  product = np.multiply(im1.data, im2.data)
  return np.sum(product)

def applyKernel(im, kernel):
  ''' return Mx, where x is im '''
  return convolve3(im, kernel)

def applyConjugatedKernel(im, kernel):
  ''' return M^T x, where x is im '''
  return convolve3(im, kernel.transpose())

def computeResidual(kernel, x, y):
  ''' return Mx - y '''
  Mx = applyKernel(x, kernel)
  out = np.subtract(Mx, y) 
  return out


def computeStepSize(r, kernel):
  alpha = (r * r) / (np.multiply(kernel.transpose, kernel) * r)
  return alpha

def deconvGradDescent(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  for i in range(niter):
    x = np.empty([im_blur.size(), im_blur.size(), 3])
    x[:, :]= 0
    if(i == 0):
      im = x
    mx = applyKernel(im_blur, kernel)
    mtmx = applyConjugatedKernel(mx, kernel)
    r = computeResidual(kernel.tranpose, im_blur, mtmx)
    alpha = computeStepSize(r, kernel)
    xi = np.add(x, np.multiply(alpha, r))
    im = dotIm(im, xi)

  return im

# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):
  return alpha

def computeConjugateDirectionStepSize(old_r, new_r):
  return beta

def deconvCG(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  return im

def laplacianKernel():
  ''' a 3-by-3 array '''
  return laplacian_kernel

def applyLaplacian(im):
  ''' return Lx (x is im)'''
  return out

def applyAMatrix(im, kernel):
  ''' return Ax, where A = M^TM'''
  return out

def applyRegularizedOperator(im, kernel, lamb):
  ''' (A + lambda L )x'''
  return out


def computeGradientStepSize_reg(grad, p, kernel, lamb):
  return alpha

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
  ''' return deblurred and regularized im '''

  return im

    
def naiveComposite(bg, fg, mask, y, x):
  ''' naive composition'''
  return out

def Poisson(bg, fg, mask, niter=200):
  ''' Poisson editing using gradient descent'''
            
  return x



def PoissonCG(bg, fg, mask, niter=200):
  ''' Poison editing using conjugate gradient '''

  return x 
  
  

#==== Helpers. Use them as possible. ==== 

def convolve3(im, kernel):
  from scipy import ndimage
  center=(0,0)
  r=ndimage.filters.convolve(im[:,:,0], kernel, mode='reflect', origin=center) 
  g=ndimage.filters.convolve(im[:,:,1], kernel, mode='reflect', origin=center) 
  b=ndimage.filters.convolve(im[:,:,2], kernel, mode='reflect', origin=center) 
  return (np.dstack([r,g,b]))

def gauss2D(sigma=2, truncate=3):
  kernel=horiGaussKernel(sigma, truncate);
  kerker=np.dot(kernel.transpose(), kernel)
  return kerker/sum(kerker.flatten())

def horiGaussKernel(sigma, truncate=3):
  from scipy import signal
  sig=signal.gaussian(2*int(sigma*truncate)+1,sigma)
  return np.array([sig/sum(sig)])



