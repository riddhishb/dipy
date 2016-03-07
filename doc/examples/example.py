
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_stanford_hardi
from dipy.data import read_sherbrooke_3shell
from dipy.denoise.noise_estimate import estimate_sigma
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise import ornlm
from dipy.denoise import ascm

def nlmeans_ornlm_comparision(S0,patch_size,block_size):
    axial = S0.shape[2]/2
    sigma = estimate_sigma(S0, N=4)
    # the nlmeans
    den = nlmeans(S0, sigma=sigma[0], mask=None,patch_radius=patch_size,block_radius=block_size)
    # the ornlm
    f=np.array(ornlm.ornlm(S0, patch_size, block_size, sigma[0]))
    plt.figure("Output of ornlm")
    plt.imshow(f[:,:,axial].T,cmap='gray')
    plt.figure("Output of nlmeans")
    plt.imshow(den[:,:,axial].T,cmap='gray')  
    plt.figure("Difference")
    plt.imshow(np.abs(den[:,:,axial].T - f[:,:,axial].T),cmap='gray')
    error = np.sum(np.abs(den.T - f.T))/np.sum(f.T)
    print(error)
    return [f,den]

def ascm_comparision(S0,A1,A2,B1,B2):
    axial = S0.shape[2]/2 
    sigma = estimate_sigma(S0,N=4)
    filterd_ornlm=np.array(ascm.ascm(S0,A1,A2,sigma[0]))#this is reported to have the top performer
    filterd_nlmeans=np.array(ascm.ascm(S0,B1,B2,sigma[0]))#this is reported to have the top performer   
    plt.figure("Output of nlmeans+ascm")
    plt.imshow(filterd_nlmeans[:,:,axial].T,cmap='gray')
    plt.figure("Output of ornlm+ascm")
    plt.imshow(filterd_ornlm[:,:,axial].T,cmap='gray')
    plt.figure("Error Image")
    plt.imshow(np.abs(filterd_nlmeans[:,:,axial].T - filterd_ornlm[:,:,axial].T),cmap='gray')
    rmsd_error = np.sum((filterd_nlmeans.T - filterd_ornlm.T)**2)/np.sum((filterd_ornlm.T)**2)
    print(rmsd_error)  

if __name__=='__main__':
    
    # fetch_sherbrook_3shell()
    img, gtab = read_sherbrooke_3shell()
    data = img.get_data()
    mask = data[..., 0] > 80
    S0 = data[..., 1].astype(np.float64)
    axial = data.shape[2]/2
    plt.imshow(S0[:,:,axial].T,cmap='gray')

    #Note: in P. Coupe et al. the rician noise was simulated as 
    #sqrt((f+x)^2 + (y)^2) where f is the pixel value and x and y are 
    #independent realizations of a random variable with Normal distribution, 
    #with mean=0 and standard deviation=h. The user must tune the 'h' 
    #parameter taking that into consideration
    # h=0.01*mv
    # We estimate the variance of the noise present using estimate_sigma function

    ################# Test 2: Comparision between the nlmeans and the ornlm ################

    [f1,den1] = nlmeans_ornlm_comparision(S0,3,5)

    ################# Test 3: nlmeans+ascm and ornlm+ascm comparision ######################

    sigma = estimate_sigma(S0,N=4)
    f2=np.array(ornlm.ornlm(S0, 3, 3, sigma[0]))
    den2 = nlmeans(S0, sigma=sigma[0], mask=None,patch_radius=3,block_radius=3)
    ascm_comparision(S0,f2,f1,den2,den1)
    plt.show()
