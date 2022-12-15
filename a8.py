import numpy as np
from utils import imageIO as io

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
  product = im1 * im2
  return np.sum(product)

def applyKernel(im, kernel):
  ''' return Mx, where x is im '''
  return convolve3(im, kernel)

def applyConjugatedKernel(im, kernel):
  ''' return M^T x, where x is im '''
  return convolve3(im, kernel.transpose())

def computeResidual(kernel, x, y):
  ''' return Mx - y '''
  Mx = convolve3(x, kernel)
  out = y - Mx
  return out


def computeStepSize(r, kernel):
  ''' alpha = r dot r / (r dot (M^T * M * r)) '''
  rdotr = dotIm(r,r)
  Mr = applyKernel(r, kernel)
  MtMr = applyConjugatedKernel(Mr, kernel)
  alpha = rdotr / dotIm(r, MtMr)
  return alpha

def deconvGradDescent(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  x = io.constantIm(im_blur.shape[0], im_blur.shape[1], 0.0)
  for i in range(niter):
    '''r = M^Ty - M^TMx'''
    r = applyConjugatedKernel(computeResidual(kernel, x, im_blur), kernel)
    alpha = computeStepSize(r, kernel)
    x = x + np.multiply(alpha,r)

  return x

# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):
  ''' write A as MtM '''
  ''' alpha = r dot r / (d dot (M^T * M * d)) '''
  rdotr = dotIm(r, r)
  Md = applyKernel(d, kernel)
  MtMd = applyConjugatedKernel(Md, kernel)
  alpha = rdotr / dotIm(d, MtMd)
  return alpha

def computeConjugateDirectionStepSize(old_r, new_r):
  '''new_r dot new_r / old_r dot old_r'''
  new_rdotr = dotIm(new_r, new_r)
  old_rdotr = dotIm(old_r, old_r)
  beta = new_rdotr / old_rdotr
  return beta

def deconvCG(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  x = io.constantIm(im_blur.shape[0], im_blur.shape[1], 0.0)
  r = applyConjugatedKernel(computeResidual(kernel, x, im_blur), kernel)
  d = r
  for i in range(niter):
    '''r = M^Ty - M^TMx'''
    alpha = computeGradientStepSize(r, d, kernel)
    new_r = r - np.multiply(alpha, applyConjugatedKernel(applyKernel(d, kernel), kernel))
    beta = computeConjugateDirectionStepSize(r, new_r)
    x = x + np.multiply(alpha,d)
    d = new_r + np.multiply(beta, d)
    r = new_r
  return x

def laplacianKernel():
  ''' a 3-by-3 array '''
  laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
  return laplacian_kernel

def applyLaplacian(im):
  ''' return Lx (x is im)'''
  laplacian_kernel = laplacianKernel()
  out = convolve3(im, laplacian_kernel)
  return out

def applyAMatrix(im, kernel):
  ''' return Ax, where A = M^TM'''
  Mx = applyKernel(im, kernel)
  out = applyConjugatedKernel(Mx, kernel)
  return out

def applyRegularizedOperator(im, kernel, lamb):
  ''' (A + lambda L )x = Ax + lambda * Lx'''
  Ax = applyKernel(im, kernel)
  L = applyLaplacian(im)
  Lx = np.multiply(lamb, L)
  return Ax + Lx


def computeGradientStepSize_reg(grad, p, kernel, lamb):
  alpha = dotIm(grad, grad) / dotIm(p, applyRegularizedOperator(p, kernel, lamb))
  return alpha

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
  ''' return deblurred and regularized im '''
  x = io.constantIm(im_blur.shape[0], im_blur.shape[1], 0.0)
  r = applyKernel(im_blur, kernel) - applyRegularizedOperator(x,kernel,lamb)
  d = r
  for i in range(niter):
    alpha = computeGradientStepSize_reg(r, d, kernel, lamb)
    x = x + np.multiply(alpha, d)
    im = x 

    '''(A + lamba * L)x'''
    b = applyRegularizedOperator(d, kernel, lamb)

    '''r - alpha * b'''
    r1 = r - np.multiply(alpha, b)

    '''beta = new_r dot new_r / old_r dot old_r'''
    beta = computeConjugateDirectionStepSize(r, r1)
    d = r1 + np.multiply(beta, d)
    r = r1
  return im

    
def naiveComposite(bg, fg, mask, y, x):
  ''' naive composition'''
  out = bg.copy()
  mask_complement = 1 - mask

  out[y:y+fg.shape[0], x:x+fg.shape[1]] *= mask_complement

  fg[mask == 0] = 0 
  bg[y : y + fg.shape[0], x : x + fg.shape[1]] = fg

  out[y:y+fg.shape[0], x:x+fg.shape[1]] += bg[y:y+fg.shape[0], x:x+fg.shape[1]]

  return out

def Poisson(bg, fg, mask, niter=200):
  ''' Poisson editing using gradient descent'''
  b = applyLaplacian(fg)
  x = (1 - mask) * bg

  for i in range(niter):
    r = b - applyLaplacian(x)

    '''copy the mask area into r'''
    r *= mask 

    '''calculate alpha r dot r / r dot Lr'''
    alpha = dotIm(r,r) / dotIm(r, applyLaplacian(r))

    '''update x with alpha and r'''
    x = x + np.multiply(alpha, r)

            
  return x



def PoissonCG(bg, fg, mask, niter=200):
  ''' Poison editing using conjugate gradient '''
  b = applyLaplacian(fg)
  x = (1 - mask) * bg
  r = b - applyLaplacian(x)
  r *= mask
  d = r

  for i in range(niter):
    '''calculate alpha r dot r / r dot Lr'''
    alpha = dotIm(r,r) / dotIm(d, applyLaplacian(d))

    '''update x with alpha and d'''
    x = x + np.multiply(alpha, d)

    r1 = r - np.multiply(alpha, applyLaplacian(d))

    '''composited derivated directions'''
    r1 *= mask 
    
    '''update the step'''
    beta = computeConjugateDirectionStepSize(r, r1)

    d = r1 + np.multiply(beta, d)
    r = r1
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



