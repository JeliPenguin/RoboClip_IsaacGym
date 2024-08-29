import numpy as np
import av
import matplotlib.pyplot as plt
import torch
import os
import PIL
from PIL import Image

def read_gif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    
    Read images from an animated GIF file.  Returns a list of numpy 
    arrays, or, if asNumpy is false, a list if PIL images.
    
    """
    
    # Load file using PIL
    pilIm = PIL.Image.open(filename)    
    pilIm.seek(0)
    
    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert() # Make without palette
            a = np.asarray(tmp)
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell()+1)
    except EOFError:
        pass
    
    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:            
            images.append( PIL.Image.fromarray(im) )
    
    # Done
    return np.array(images)

def readVideoPyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
def read_video(filename,resolution):

    container = av.open(filename)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / resolution).astype(int)
    frames = readVideoPyav(container, indices)

    return frames,total_frames

def crop_to_dim(frames,toHeight,toWidth):
    
    _,height,width,_ = frames.shape
    center = max(toHeight,height)/2, max(toWidth,width)/2
    x = int(center[1] - toWidth/2)
    y = int(center[0] - toHeight/2)
    frames = np.array([frame[y:y+toHeight, x:x+toWidth] for frame in frames])
    return frames

def visualize_frames(frames):
    for i in range(len(frames)):
        print(frames[i].shape)
        plt.imshow(frames[i])
        plt.title(f"Frame {i}")
        plt.show()

def compute_similarity(output_embedding,target_embedding):
    similarity_matrix = torch.matmul(target_embedding, output_embedding.t())
    
    reward = similarity_matrix.detach().numpy()[0][0]

    return reward