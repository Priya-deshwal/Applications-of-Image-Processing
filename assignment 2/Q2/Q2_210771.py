import cv2
import numpy as np

def gauss(img,spatialKern, rangeKern):
    gaussianSpatial = 1 / np.sqrt(2*np.pi* (spatialKern**2)) #gaussian function to calcualte the spacial kernel ( the first part 1/sigma * sqrt(2π))
    gaussianRange= 1 / np.sqrt(2*np.pi* (rangeKern**2)) #gaussian function to calcualte the range kernel
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussianRange)
    xx=-spatialKern + np.arange(2 * spatialKern + 1)
    yy=-spatialKern + np.arange(2 * spatialKern + 1)
    x, y = np.meshgrid(xx , yy )
    spatialGS = gaussianSpatial*np.exp(-(x **2 + y **2) /(2 * (gaussianSpatial**2) ) ) #calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2)
    return matrix,spatialGS

def jointBilateralFilter(img, img1,spatialKern, rangeKern):
    h, w, ch = img.shape #get the height,width and channel of the image with no flash
    #orgImg = padImage(img,spatialKern) #pad image with no flash
    orgImg = np.pad(img, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')
    secondImg = np.pad(img1, ((spatialKern, spatialKern), (spatialKern, spatialKern), (0, 0)), 'symmetric')
    matrix,spatialGS=gauss(img,spatialKern, rangeKern) #apply gaussian function

    outputImg = np.zeros((h,w,ch), np.uint8) #create a matrix the size of the image
    summ=1
    for x in range(spatialKern, spatialKern + h):
        for y in range(spatialKern, spatialKern + w):
            for i in range (0,ch): #iterate through the image's height, width and channel
                #apply the equation that is mentioned in the pdf file
                neighbourhood=secondImg[x-spatialKern : x+spatialKern+1 , y-spatialKern : y+spatialKern+1, i] #get neighbourhood of pixels
                central=secondImg[x, y, i] #get central pixel
                res = matrix[ abs(neighbourhood - central) ]  # subtract them
                summ=summ*res*spatialGS #multiply them with the spatial kernel
                norm = np.sum(res) #normalization term
                outputImg[x-spatialKern, y-spatialKern, i]= np.sum(res*orgImg[x-spatialKern : x+spatialKern+1, y-spatialKern : y+spatialKern+1, i]) / norm # apply full equation of JBF(img,img1)
    return outputImg

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    #image = cv2.imread(image_path_b)
    img1 = cv2.imread(image_path_a)
    img2 = cv2.imread(image_path_b)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  

    ycbcr_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    ycbcr_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)

# Specify the interpolation parameter (alpha)
    alpha = 0.4  # You can adjust this value to control the blending

# Linearly interpolate between the two images
    blended_image = cv2.addWeighted(ycbcr_image1, 1 - alpha, ycbcr_image2, alpha, 0)
    blended_image = cv2.cvtColor(blended_image, cv2.COLOR_YCrCb2BGR)
    img2 = blended_image

    sig_col = 30

    flash =img2
    no_flash = img1

    sigma = (((flash.shape[0]**2)+(flash.shape[1]**2))**0.5)*0.025

    base_f = jointBilateralFilter(flash, flash, sig_col, sigma)
    base_nf = jointBilateralFilter(no_flash, flash,30, sigma)

    flash = flash.astype('float')
    base_f = base_f.astype('float')

    flash = cv2.add(flash,0.02)
    base_f = cv2.add(base_f,0.02)
    detail = cv2.divide(flash, base_f)

    base_nf = base_nf.astype('float')
    intensity = cv2.multiply(base_nf, detail)

    no_flash = no_flash.astype('float')
    no_flash = cv2.resize(no_flash, (no_flash.shape[1], no_flash.shape[0]))
    #n_flsh = np.array(no_flash)
    b_no_flash, g_no_flash, r_no_flash = cv2.split(no_flash)
    nflash_color = img2
    nflash_color = nflash_color.astype('float')
    b = nflash_color[:, :, 0]
    g = nflash_color[:, :, 1]
    r = nflash_color[:, :, 2]

    b = cv2.resize(b, (no_flash.shape[1], no_flash.shape[0]))
    g = cv2.resize(g, (no_flash.shape[1], no_flash.shape[0]))
    r = cv2.resize(r, (no_flash.shape[1], no_flash.shape[0]))

    b = cv2.divide(b, b_no_flash)
    g = cv2.divide(g, g_no_flash)
    r = cv2.divide(r, r_no_flash)

    intensity=intensity.astype('float')
    #plt.imshow(intensity)
    b_intensity, g_intensity, r_intensity = cv2.split(intensity)
    b = cv2.multiply(b, b_intensity)
    g = cv2.multiply(g, g_intensity)
    r = cv2.multiply(r, r_intensity)

    # result = np.zeros((flash.shape[0], flash.shape[1],3), np.uint8)
    # result[:, :, 0] = b
    # result[:, :, 1] = g
    # result[:, :, 2] = r

    # cv2.imwrite('result.jpg', result)
    result = cv2.merge([b, g, r])
    return result
