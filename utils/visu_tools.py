import numpy as np
def superimpose_mask(image_array, mask_array, opacity=1, color_index=0, grayscale=True):
    '''
    superimpose the binary mask on the RGB image with adjustable opacity level
    color index = 0, 1 and 2 indicates color red, green and blue in order
    '''
    if grayscale:
        superimposed = gray_to_colored(image_array)
    else:
        superimposed = image_array.copy()
        
    colored_mask = np.zeros(image_array.shape)
    colored_mask[:, :,color_index] = mask_array[:,:,0] == 1
    colored_mask = colored_mask.astype(np.bool)
    superimposed[colored_mask] = opacity * 1 + (1 - opacity) * superimposed[colored_mask]
    return superimposed
