from S3D_HowTo100M.s3dg import S3D
import torch
from VLMs.utils import read_video,compute_similarity,read_gif, visualize_frames,crop_to_dim
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
import os

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

class VLM():
    def __init__(self,method) -> None:
        print("Running ",method)

class S3D_Method(VLM):
    def __init__(self) -> None:
        super().__init__("S3D")
        self.model = S3D(os.path.join(ABS_PATH,'s3d_dict.npy'), 512)
        self.model.load_state_dict(torch.load(os.path.join(ABS_PATH,'s3d_howto100m.pth')))
        self.model = self.model.eval()
    
    def _preprocess_video(self,frames):
        # frames = readGif(video_path)
        # print(frames.shape)
        frames = crop_to_dim(frames,250,250)
        # print(frames.shape) 
        frames = frames[None, :,:,:,:]

        frames = frames.transpose(0, 4, 1, 2, 3)

        if frames.shape[1]>3:
            frames = frames[:,:3]

        video = torch.from_numpy(frames)

        return video.float()
    
    
    def _preprocess_eureka(self,frames):
        frames = crop_to_dim(frames,250,250) 
        #visualize_frames(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)

        video=torch.from_numpy(frames)

        return video.float()

    
    def embed_video(self,frames):
        
        video = self._preprocess_video(frames)

        return self.model(video.float())['video_embedding']
    
    def embed_text(self,input_text):

        return self.model.text_module(input_text)['text_embedding']


class Video_Llava_Method(VLM):
    def __init__(self) -> None:
        super().__init__("Video Llava")
        self.model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="cuda")
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    
    def inference(self,video_path):

        video = read_video(video_path,resolution=8)

        user_prompt = input("Enter a prompt for the video: ")
        prompt = f"USER: <video>{user_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt")

        # Move the inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(**inputs, max_new_tokens=200)
        decoded_output = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return decoded_output

class XCLIP_Method(VLM):
    def __init__(self) -> None:
        super().__init__("XCLIP")

    

class GIT_Method(VLM):
    def __init__(self) -> None:
        super().__init__("GIT")

def runVLMTest(model):
    test_inputs = [
        # ["video/dog.mp4","Dog Running"],
        ["video/button-human.gif","Pressing a button with a single finger"],
        ["video/human_opening_door.gif","Human opening a fridge door with his right hand"],
        ["video/drawer-open-human2.gif","Human opening a drawer"]
    ]

    for input in test_inputs:
        video_path = input[0]
        prompt = input[1]

        a = model.embed_video(video_path)

        b = model.embedText(prompt)

        print(compute_similarity(a,b))
    
    c = model.embed_video("video/human_opening_door.gif")
    d = model.embed_video("video/drawer-open-human2.gif")
    e = model.embed_video("video/button-human.gif")
    print(compute_similarity(c,d))
    print(compute_similarity(d,e))

if __name__ == "__main__":
    video_path = os.path.join(ABS_PATH,"./video/new.mp4")
    #s3d = S3D_Method()
    #s3d.embed_video(video_path)
    # frames = read_video()
    #pre_processed = s3d._preprocess_eureka(frames)


    #runVLMTest(s3d)
    vLlava = Video_Llava_Method()
    print(vLlava.inference(video_path))