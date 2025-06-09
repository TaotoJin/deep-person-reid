from torchreid.utils import FeatureExtractor
import torch 
import openvino as ov 
import openvino.properties as props
import os 
import numpy as np 

from pathlib import Path 


MODEL_ZOO_DIR = Path(os.path.join(os.path.dirname(__file__), 'model_zoo'))


# implement same style as LATransformer 
class TorchREID:
    def __init__(self, model_path=None, device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device      
        
        self.model = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path= MODEL_ZOO_DIR / 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
            device='cpu'
        )
        
    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = [image]
            
        features = self.model(image)
        features_norm = torch.nn.functional.normalize(features, dim=1)
        return features_norm
    
    def predict_batch(self, images):
        if isinstance(images, np.ndarray):
            images = [images]
        
        features = self.model(images)
        features_norm = torch.nn.functional.normalize(features, dim=1)
        return features_norm
    
class TorchREIDOpenvino(TorchREID):
    def __init__(self, model_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device 
        self.model = OpenvinoFeatureExtractor(
            model_name='osnet_x1_0',
            model_path= MODEL_ZOO_DIR / 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
            device='cpu'
        )
        

        
class OpenvinoFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ov_model = self.openvino_osnet_setup()
        
    def openvino_osnet_setup(self):
        torch.manual_seed(42)
        dummy_input = torch.randn(1, 3, 256, 128)
        ov_model = ov.convert_model(self.model, example_input=dummy_input)
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, device_name="GPU")

        return compiled_model
    def __call__(self, input):
        
        # If needed, you can still call the original __call__ method
        features = super().__call__(input)
        
        
        
        return features
    
    
if __name__ == '__main__':
    import cv2 
    
    path_ = Path(os.path.join(os.path.dirname(__file__)))
    image1 = cv2.imread(path_ /'assets/bella_goth1.jpg')
    image2 = cv2.imread(path_ /'assets/bella_goth2.jpg')
    image3 = cv2.imread(path_/'assets/mortimer_goth1.jpg')

    # model = TorchREID()
    model = TorchREIDOpenvino()
    output = model.predict([image1])
    print(output.shape)
    # For batch prediction
    images = [image2, image3]  # Example with the same image twice
    batch_output = model.predict_batch(images)
    print(batch_output.shape)    
    
    similarity = torch.cosine_similarity(output, batch_output, dim=1)
    print("Similarity scores:", similarity)     
    
