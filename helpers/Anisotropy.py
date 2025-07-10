from numpy import ma
import numpy as np
from scipy.linalg import eig
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

def Anisotropy(Dataset, return_rotation = False, one_sonic = False):

    '''Works with xarray dataset with Reynolds stress tensor labels:
    #  uu,vv,ww,ecc..
    maybe problem with 1 height only meas
    returns the barycentric coordinates and RGB values
    returns isotropic data where nans are found
    '''
    
    #Filter nans to value 0 beacuse not handled by linalg.eig
    Dataset = Dataset.where(np.isfinite(Dataset.uu), other=0)
    if not one_sonic:
        (length,levels) = np.shape(Dataset['uu'])
    
        #Compute Reynolds tensor
        Reynolds_tensor = np.zeros((3,3,length,levels))
        trace=Dataset['uu'] + Dataset['vv'] + Dataset['ww']
    
        Reynolds_tensor[0,0] = Dataset['uu'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[0,1] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,0] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,1] = Dataset['vv'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[1,2] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,1] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[0,2] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,0] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,2] = Dataset['ww'].data/ma.masked_equal(trace,0) - 1/3
        
        #Compute eigenvalues
        eigenvalues = np.zeros((3,length,levels))
        eigenvectors = np.zeros((3,3, length, levels))
        for n in range(length):
            for l in range(0,levels):
                #compute
                eigval, eigvec = eig(Reynolds_tensor[:,:,n,l])
                #sort
                idx = eigval.argsort()[::-1]
                eigenvalues[:,n,l] = eigval[idx]
                eigenvectors[:, :, n, l] = eigvec[:, idx]
        
        #choose system of eigenvectors (not necessary)
        eigenvectors[:,0] = np.where(eigenvectors[0,0]>0, eigenvectors[:,0], -eigenvectors[:,0])
        eigenvectors[:,2] = np.where(eigenvectors[2,2]>0, eigenvectors[:,2], -eigenvectors[:,2])
        eigenvectors[:,1] = np.cross(eigenvectors[:,2],eigenvectors[:,0],axis=0)
    
        #check that the sum of eigenvalues is 0
        sum_eig=np.sum(eigenvalues)
        if  sum_eig > 0.000001 :
            print("Warning! the sum of all the eigenvalues is {}".format(sum_eig))
            
        #Compute barycentric map
        barycentric=np.zeros((2,length,levels))
        barycentric[0] = eigenvalues[0] - eigenvalues[1] + 1.5*eigenvalues[2] + 0.5
        barycentric[1] = np.sqrt(3)/2*(3*eigenvalues[2]+1)
        
        #Color code (Emori and Jaccarino 2014)
        RGB=np.zeros((3,length,levels))
        RGB[0]=eigenvalues[0]-eigenvalues[1]
        RGB[2]=2*(eigenvalues[1]-eigenvalues[2])
        RGB[1]=3*eigenvalues[2]+1
        RGB=np.moveaxis(RGB,0,-1)

    else:
        
        length = len(Dataset['uu'])
    
        #Compute Reynolds tensor
        Reynolds_tensor = np.zeros((3,3,length))
        trace=Dataset['uu'] + Dataset['vv'] + Dataset['ww']
    
        Reynolds_tensor[0,0] = Dataset['uu'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[0,1] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,0] = Dataset['uv'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[1,1] = Dataset['vv'].data/ma.masked_equal(trace,0) - 1/3
        Reynolds_tensor[1,2] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,1] = Dataset['vw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[0,2] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,0] = Dataset['uw'].data/ma.masked_equal(trace,0)
        Reynolds_tensor[2,2] = Dataset['ww'].data/ma.masked_equal(trace,0) - 1/3
        
        #Compute eigenvalues
        eigenvalues = np.zeros((3,length))
        eigenvectors = np.zeros((3,3, length))
        for n in range(length):
            #compute
            eigval, eigvec = eig(Reynolds_tensor[:,:,n])
            #sort
            idx = eigval.argsort()[::-1]
            eigenvalues[:,n] = eigval[idx]
            eigenvectors[:, :, n] = eigvec[:, idx]
        
        #choose system of eigenvectors (not necessary)
        eigenvectors[:,0] = np.where(eigenvectors[0,0]>0, eigenvectors[:,0], -eigenvectors[:,0])
        eigenvectors[:,2] = np.where(eigenvectors[2,2]>0, eigenvectors[:,2], -eigenvectors[:,2])
        eigenvectors[:,1] = np.cross(eigenvectors[:,2],eigenvectors[:,0],axis=0)
    
        #check that the sum of eigenvalues is 0
        sum_eig=np.sum(eigenvalues)
        if  sum_eig > 0.000001 :
            print("Warning! the sum of all the eigenvalues is {}".format(sum_eig))
            
        #Compute barycentric map
        barycentric=np.zeros((2,length))
        barycentric[0] = eigenvalues[0] - eigenvalues[1] + 1.5*eigenvalues[2] + 0.5
        barycentric[1] = np.sqrt(3)/2*(3*eigenvalues[2]+1)
        
        #Color code (Emori and Jaccarino 2014)
        RGB=np.zeros((3,length))
        RGB[0]=eigenvalues[0]-eigenvalues[1]
        RGB[2]=2*(eigenvalues[1]-eigenvalues[2])
        RGB[1]=3*eigenvalues[2]+1
        RGB=np.moveaxis(RGB,0,-1)
    
    if return_rotation:
        return barycentric, RGB, eigenvalues, eigenvectors
    return barycentric, RGB
