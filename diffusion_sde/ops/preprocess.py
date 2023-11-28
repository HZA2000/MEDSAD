from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip
from diffusion_sde.configs.config import CFGS 

img_size = CFGS["model"]["image_size"]


train_transform = Compose([
                            Resize((img_size, img_size)),
                            ToTensor(),
                            RandomHorizontalFlip(p=0.25),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]
                        )

val_transform = Compose([
                            Resize((img_size, img_size)),
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]
                        )
