from cgi import test
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch 
from torchvision.utils import make_grid, save_image
from diffusion_sde.configs.config import CFGS
from diffusion_sde.models.ema import ExponentialMovingAverage
# from diffusion_sde.models.unet import UNet
# from diffusion_sde.models.unet_fft import UNet
# from diffusion_sde.models.unet_fft_plus import UNet
# from diffusion_sde.models.unet_fft_plusv2 import UNet
from diffusion_sde.models.unet_fft_plusv3 import UNet
# from diffusion_sde.models.unet_wavelet import UNet
# from diffusion_sde.models.unet_waveletv2 import UNet
from diffusion_sde.ops.sde import VPSDE
from diffusion_sde.ops.loss import get_sde_loss_fn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusion_sde.ops.sampling import get_sampler, get_guide_sampler
from diffusion_sde.ops.preprocess import train_transform, val_transform 
from diffusion_sde.datasets import Braindatasets, DatasetFromSubset
 
import time
import SimpleITK as sitk
import cv2
from imageNamesBrain import test_img_list as BUSI_test_img_list
from imageNamesBrain import test_mask_list as BUSI_test_mask_list

# Initialize the values and the functions                           
VAL_FRAC = CFGS["training"]["val_frac"]
BATCH_SIZE = CFGS["training"]["batch_size"]
N_ITERS = CFGS["training"]["n_iters"]
CKPT_FREQ = CFGS["training"]["ckpt_freq"]
N_STEPS = CFGS["sampling"]["n_steps"]
DEVICE = CFGS["device"]
BATCH_SIZES = [1,2,4,8,16,32,40,48,64]

MODEL = UNet(**CFGS["model"])
SDE = VPSDE(N=N_STEPS)
LOSS_FN = get_sde_loss_fn(sde=SDE, device=DEVICE)
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=CFGS["optim"]["lr"], betas=(CFGS["optim"]["beta1"], 0.999), 
                       eps=CFGS["optim"]["eps"], weight_decay=CFGS["optim"]["weight_decay"])


# Setup the diffSDE class
class diffSDE(object):
    def __init__(self, sde=SDE, model=MODEL, loss_fn=LOSS_FN, optimizer=OPTIMIZER, config=CFGS):
        """Class to train the model and generate samples

        Args:
            sde (optional): SDE instance. Defaults to SDE.
            model (optional): preferred model for the score function. Defaults to MODEL.
            loss_fn (optional): the loss fn to train on. Defaults to LOSS_FN.
            optimizer (optional): the desired optimizer. Defaults to OPTIMIZER.
            config (optional): Configs files. Defaults to CFGS.
        """
        
        # Set the device and the dataset as class attributes
        self.config = config
        self.sde = sde
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = self.config["device"]
        
        # Set the device and send the models to the device
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(model).to(self.device)
        else:
            self.model = model
            
        # Set the data loaders and writer
        self.train_loader = None
        self.val_loader = None
        self.total_epochs = 0
        
        # Set the states of the two models
        self.state = self.get_state(self.model, self.optimizer)
        
        # Internal variables
        self.losses = []
        self.val_losses = []
        
        # Set the optimizer function and the training/evaluation step function
        self.optimize_fn = self._optimization_manager()
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

        self.sample_num = 0
                
    def set_loaders(self, dataset, stop_num=np.inf, val_frac=VAL_FRAC, batch_size=BATCH_SIZE):
        """Set the data loaders for training/evaluation.

        Args:
            dataset: the dataset for training and validation
            val_frac (float, optional): fraction of the dataset for validation. Defaults to VAL_FRAC.
            batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        """
        
        # Assert that some of the dataset is always set aside for validation
        # assert isinstance(dataset, datasets), "val_frac must be more than 0 and less than 1"
        
        # Assert that the dataset is always an instance of datasets class
        assert 0 < val_frac < 1, "val_frac must be more than 0 and less than 1" 
        
        # Assert that the batch_size is always 1,2,4,8 or 16
        assert batch_size in BATCH_SIZES, "Allowed batch sizes are 1,2,4,8,16"
        
        # Split the dataset into training and testing set
        indices = torch.randperm(len(dataset)).tolist()
        train_len = len(dataset) - int(len(dataset)*0.1)
        train_subset = torch.utils.data.Subset(dataset, indices[:train_len])
        val_subset = torch.utils.data.Subset(dataset, indices[train_len:])
        
        # train_dataset = DatasetFromSubset(train_subset, train_transform)
        # val_dataset = DatasetFromSubset(val_subset, val_transform)

        train_dataset = DatasetFromSubset(train_subset)
        val_dataset = DatasetFromSubset(val_subset)
        
        # Set the train and validation loaders
        self.train_loader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers = 4,
                                    pin_memory=True,
                                    drop_last=False,
                                    shuffle=True
                                )
        
        self.val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    num_workers = 4,
                                    pin_memory=True,
                                    drop_last=False,
                                )
        self.test_img_list, self.test_mask_list = self.get_test_set(stop_num=stop_num, resize=CFGS['model']['image_size'])

    def get_state(self, model, optimizer):
        """Set the current state of the parameters

        Args:
            model: the neural network architecture for the score function.
            optimizer: the optimizer used for the optimization.

        Returns:
            dict
        """
        ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config["ema"]["ema_rate"])
        state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
        return state
    
    def _optimization_manager(self):
        """Returns an optimize_fn based on `config`."""

        def optimize_fn(optimizer, params, step, lr=self.config["optim"]["lr"],
                        warmup=self.config["optim"]["warmup"],
                        grad_clip=self.config["optim"]["grad_clip"]):
            
            """Optimizes with warmup and gradient clipping (disabled if negative)."""
            
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)

            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

        return optimize_fn
    
    def _make_train_step_fn(self):
        """Builds function that performs a step in the training loop"""
        
        def train_step_fn(batch):
            """Running one step of training.

            Args:
                batch: A mini-batch of training data.

            Returns:
                The average loss value of the mini-batch.
            """
            model = self.state["model"]
            optimizer = self.state["optimizer"]
            model.train()
            optimizer.zero_grad()
            loss = self.loss_fn(model, batch)
            loss.backward()
            self.optimize_fn(optimizer, model.parameters(), 
                                step=self.state["step"])
            self.state["step"] += 1
            self.state["ema"].update(model.parameters())
               
            return loss

        return train_step_fn
    
    def _make_val_step_fn(self):
        """Builds function that performs a step in the validation loop"""
        
        
        def perform_val_step_fn(batch):
            """Running one step of validation.

            Args:
                batch: A mini-batch of evaluation data.

            Returns:
                The average loss value of the mini-batch.
            """
            model = self.state["model"]
            ema = self.state["ema"]
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            loss = self.loss_fn(model, batch)
            ema.restore(model.parameters())
            
            return loss

        return perform_val_step_fn

    def _mini_batch_loss(self, validation=False):
        """Calculate the loss value for the mini-batch in either training or evaluation mode

        Args:
            validation (bool, optional): Set to true while training. Defaults to False.

        Returns:
            the calculated loss value
        """
        
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
            epoch_type = "Val Epoch" 
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
            epoch_type = "Train Epoch" 
            
        if data_loader is None:
            return None
            
        mini_batch_losses = []
        
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"{epoch_type}: {self.total_epochs}")
                mini_batch_loss = torch.mean(step_fn(batch))
                mini_batch_losses.append(mini_batch_loss.item())

        loss = np.mean(mini_batch_losses)
        return loss
    
    def set_seed(self, seed=42):
        """Set the seed for reproducibility

        Args:
            seed (int, optional): Defaults to 42.
        """
        if seed >= 0:
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False    
            torch.manual_seed(seed)
            random.seed(seed)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True   
            
    def train(self, n_iters=N_ITERS, ckpt_freq=CKPT_FREQ, seed=42):
        """Run the training loop over n_epochs

        Args:
            n_iters (int, optional): number of epochs. Defaults to 1001.
            ckpt_freq (int, optional): specify the checkpoint frequency.
            seed (int, optional): Defaults to 42.
        """
        
        initial_step = self.total_epochs
        self.set_seed(seed)
        
        for step in range(initial_step, n_iters):
            
            # Keep track of the number of epochs
            self.total_epochs +=1 
            
            # Training            
            loss = self._mini_batch_loss(validation=False)        
            self.losses.append(loss)
            
            # Validation
            with torch.no_grad():
                val_loss = self._mini_batch_loss(validation=True)
                self.val_losses.append(val_loss)
            
            print('step: {}, train loss: {:.5f}, val loss: {:.5f}'.format(step, loss, val_loss))
            # Save the checkpoints 
            if ckpt_freq: 
            # if 1: 
                checkpoint_dir = self.config["training"]["ckpt_dir"]
                if step != 0 and step % ckpt_freq == 0 or step == n_iters:
                # if 1:
                    t0 = time.time()
                    self.sample_val(step=step)
                    t1 = time.time()
                    print('\n')
                    print('inference time: ', t1 - t0)
                    self.save_checkpoint(checkpoint_dir)
                    # Print the current time and the number of epochs
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"Epochs Completed: {self.total_epochs}")
                    print(f"Current Time: {current_time}")    
                            
    def save_checkpoint(self, ckpt_dir):
        """Builds dictionary with all elements for resuming training

        Args:
            ckpt_dir (str): directory where the checkpoint file is located
        """
        
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
            filepath = os.path.join(ckpt_dir, "brain_checkpoint_gen_mask.pth")
            with open(filepath, 'w') as fp:
                pass
            
        saved_state = {
                    'model_state_dict':  self.state["model"].state_dict(),
                    'optimizer_state_dict':  self.state["optimizer"].state_dict(),
                    'ema_state_dict':    self.state["ema"].state_dict(),
                    'step':  self.state["step"],
                    'loss': self.losses,
                    'val_loss': self.val_losses,
                    'total_epochs': self.total_epochs,
                    }
        
        # filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_256_wavelet_ec.pth")
        # filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_128_fft_plusv2_ec.pth")
        filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_128_fft_plusv3_brain.pth")
        # filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_128_wavelet_ec.pth")
        # filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_128_brain.pth")
        # filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_128_fft_brain.pth")
        # filepath = os.path.join(ckpt_dir, "checkpoint_gen_img_early_fusion_128_waveletv2_ec.pth")
        torch.save(saved_state, filepath)

    def load_checkpoint(self, filepath):
        """Loads dictionary

        Args:
            filepath (str): directory where the checkpoint file is located
        """
        loaded_states = torch.load(filepath)

        # Restore states for models and optimizers
        self.state['model'].load_state_dict(loaded_states['model_state_dict'])
        self.state['optimizer'].load_state_dict(loaded_states['optimizer_state_dict'])
        self.state['ema'].load_state_dict(loaded_states['ema_state_dict'])
        self.state['step'] = loaded_states['step']
        
        self.total_epochs = loaded_states['total_epochs']
        self.losses = loaded_states['loss']
        self.val_losses = loaded_states['val_loss']
        
    def generate_samples(self, n_steps=N_STEPS, batch=1, save=False):
        """translates a given batch of image to another domain.
        
        Args:
            n_steps (int, optional): Number of sampling steps. Defaults to 1500
            batch (int, optional): Batch size. Defaults to 1.
            save (bool, optional): Weather to save the generated image in a folder

        Returns:
            a batch of sample images
        """
        assert n_steps > 1 and isinstance(n_steps, int), "Number of steps should be more than one." 
        
        img_size = self.config["model"]["image_size"]
        shape = torch.Size([batch, 1, img_size, img_size])
        sampling_fn = get_sampler(sde=self.sde, shape=shape, device=self.device)
        
        model = self.state["model"]
        ema = self.state["ema"]
        
        # Generate the samples
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

        samples = sampling_fn(model, num_steps=n_steps)
        
        ema.restore(model.parameters())
        
        samples = samples.detach().cpu()
        
        # Plot the samples
        self.plot_samples(samples)
        
        # Save the samples
        if save:
            self.sample_num += 1
            samples_dir = self.config["sampling"]["sample_dir"]
            if not os.path.exists(samples_dir):
                os.mkdir(samples_dir)
            
            # Save the images in the samples directory
            save_image(samples, f"{samples_dir}/samples_{self.sample_num}.png")
        
        return samples

    def get_test_set(self, resize=128, stop_num=np.inf):
        n = len(BUSI_test_img_list)
        test_img_list = []
        test_mask_list = []
        for i in range(n):
            if i + 1 > stop_num:
                break
            test_img = cv2.imread(BUSI_test_img_list[i])
            test_mask = cv2.imread(BUSI_test_mask_list[i])

            # print(test_img.shape)
            # print(test_mask.shape)
            test_img = cv2.resize(test_img, (resize, resize))
            test_mask = cv2.resize(test_mask, (resize, resize), interpolation=0)

            # print(test_img.shape)
            # print(test_mask.shape)

            test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
            test_mask = cv2.cvtColor(test_mask, cv2.COLOR_RGB2GRAY)

            test_mask = test_mask / 255.
            test_img = test_img / 255.

            test_img = test_img.astype(np.float32)
            test_mask = test_mask.astype(np.float32)

            cv2.imwrite('inference_results/raw_dir/raw_img_{}_128.png'.format(i), test_img * 255)
            cv2.imwrite('inference_results/raw_dir/raw_mask_{}_128.png'.format(i), test_mask * 255)

            test_img = torch.from_numpy(test_img).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
            test_mask = torch.from_numpy(test_mask).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
            test_img_list.append(test_img)
            test_mask_list.append(test_mask)
            # print(i + 1, stop_num)
        # exit()
        return test_img_list, test_mask_list
    
    def sample_val(self, step=0, n_steps=N_STEPS, batch=1, save=False):
        assert n_steps > 1 and isinstance(n_steps, int), "Number of steps should be more than one." 
        
        img_size = self.config["model"]["image_size"]
        shape = torch.Size([batch, 1, img_size, img_size])
        sampling_fn = get_guide_sampler(sde=self.sde, shape=shape, device=self.device)
        
        model = self.state["model"]
        ema = self.state["ema"]
        
        # Generate the samples
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        
        print('validating...')
        for test_img, test_mask in zip(self.test_img_list, self.test_mask_list):
            samples = sampling_fn(model, test_mask, num_steps=n_steps)
        
            ema.restore(model.parameters())

            print(samples.max(), samples.min())    
            samples = samples.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()

            samples = samples
            test_img = test_img.detach().cpu().numpy()
            test_mask = test_mask.detach().cpu().numpy()

            # print(test_img.shape, test_mask.shape)
            # cv2.imwrite('pred_img_{}_128_brain.png'.format(step), samples[0][0])
            # cv2.imwrite('pred_img_{}_128_fft_brain.png'.format(step), samples[0][0])
            cv2.imwrite('pred_img_{}_128_fft_plusv3_brain.png'.format(step), samples[0][0])
            # cv2.imwrite('pred_img_{}_wavelet_ec_128.png'.format(step), samples[0][0])
            # cv2.imwrite('pred_img_{}_waveletv2_ec_128.png'.format(step), samples[0][0])
            # cv2.imwrite('pred_img_{}_fft_plusv2_ec_128.png'.format(step), samples[0][0])
            cv2.imwrite('test_img.png', test_img[0][0] * 255)
            cv2.imwrite('test_mask.png', test_mask[0][0] * 255)
    
    def sample_test(self, test=False, n_steps=N_STEPS, batch=1, save=False):
        assert n_steps > 1 and isinstance(n_steps, int), "Number of steps should be more than one." 
        
        img_size = self.config["model"]["image_size"]
        shape = torch.Size([batch, 1, img_size, img_size])
        sampling_fn = get_guide_sampler(sde=self.sde, shape=shape, device=self.device)
        
        model = self.state["model"]
        ema = self.state["ema"]
        
        # Generate the samples
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        
        test_img_list, test_mask_list = self.get_test_set()
        print('inferce ing...')
        for num, (test_img, test_mask) in enumerate(zip(test_img_list, test_mask_list)):
            t0 = time.time()
            samples = sampling_fn(model, test_mask, num_steps=n_steps)
        
            ema.restore(model.parameters())

            print(samples.max(), samples.min())    
            samples = samples.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()


            # samples = samples.detach().cpu()
            test_img = test_img.detach().cpu().numpy()
            test_mask = test_mask.detach().cpu().numpy()
            
            
            cv2.imwrite('inference_results/baseline_dir/inference_img_{}_128.png'.format(num), samples[0][0])
            # cv2.imwrite('inference_results/fft_dir/inference_img_fft_{}_128.png'.format(num), samples[0][0])
            # print(1111111)
            # cv2.imwrite('inference_resuts/inference_img_{}_128.png'.format(num), samples[0][0])
      
            # cv2.imwrite('inference_resuts/raw_img_{}_128.png'.format(num), test_img[0][0] * 255)
            # cv2.imwrite('inference_resuts/raw_mask_{}_128.png'.format(num), test_mask[0][0] * 255)
            t1 = time.time()
            print(num, ' inference time: ', t1 - t0)

    def get_synthesis_set(self, resize=128, stop_num=np.inf):
        n = len(BUSI_test_img_list)
        test_img_list = []
        test_mask_list = []
        for i in range(n):
            if i + 1 > stop_num:
                break
            test_img = cv2.imread(BUSI_test_img_list[i])
            test_mask = cv2.imread(BUSI_test_mask_list[i])

            # print(test_img.shape)
            # print(test_mask.shape)
            test_img = cv2.resize(test_img, (resize, resize))
            test_mask = cv2.resize(test_mask, (resize, resize), interpolation=0)

            # print(test_img.shape)
            # print(test_mask.shape)

            test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
            test_mask = cv2.cvtColor(test_mask, cv2.COLOR_RGB2GRAY)

            test_mask = test_mask / 255.
            test_img = test_img / 255.

            test_img = test_img.astype(np.float32)
            test_mask = test_mask.astype(np.float32)
            # test_img = torch.from_numpy(test_img).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
            # test_mask = torch.from_numpy(test_mask).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
            test_img_list.append(test_img)
            test_mask_list.append(test_mask)
            # print(i + 1, stop_num)
            
        return test_img_list, test_mask_list

    def sample_synthesis(self, test=False, n_steps=N_STEPS, batch=1, save=False):
        assert n_steps > 1 and isinstance(n_steps, int), "Number of steps should be more than one." 
        
        def rotate_mask(mask, angle):
            height, width = mask.shape
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_mask = cv2.warpAffine(mask.astype(np.uint8), rotation_matrix, (width, height))
            return rotated_mask.astype(np.int8)

        def shift_mask(mask, shift_x, shift_y):
            shifted_mask = np.roll(mask, shift_x, axis=0)
            shifted_mask = np.roll(shifted_mask, shift_y, axis=1)
            return shifted_mask

        def gen_mask_based_on_reference_mask(mask):
            x = np.random.randint(low=-10, high=10)
            y = np.random.randint(low=-10, high=10)
            angle = np.random.randint(low=-10, high=10)
            new_mask = rotate_mask(shift_mask(mask, x, y), angle)
            return new_mask

        img_size = self.config["model"]["image_size"]
        shape = torch.Size([batch, 1, img_size, img_size])
        sampling_fn = get_guide_sampler(sde=self.sde, shape=shape, device=self.device)
        
        model = self.state["model"]
        ema = self.state["ema"]
        
        # Generate the samples
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        
        test_img_list, test_mask_list = self.get_synthesis_set()
        print('inferce ing...')
        syn_num = 6
        count = 194
        for num, (test_img, test_mask) in enumerate(zip(test_img_list, test_mask_list)):
            t0 = time.time()
            test_mask_np = test_mask
            for _ in range(syn_num):
                test_mask = gen_mask_based_on_reference_mask(test_mask_np)
                test_mask = torch.from_numpy(test_mask).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
                samples = sampling_fn(model, test_mask, num_steps=n_steps)
            
                ema.restore(model.parameters())
                
                print(samples.max(), samples.min())    
                samples = samples.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()


                # samples = samples.detach().cpu()
                # test_img = test_img.detach().cpu().numpy()
                # test_mask = test_mask.detach().cpu().numpy()

                # cv2.imwrite('inference_img_{}_wavelet_ec.png'.format(num), samples[0][0])
                # cv2.imwrite('inference_img_{}_fft_ec_128.png'.format(num), samples[0][0])
                
                # cv2.imwrite('inference_resuts/inference_img_fft_{}_128.png'.format(count), samples[0][0])
                cv2.imwrite('inference_resuts/fft_plus_dir/inference_img_fft_plus_{}_128.png'.format(count), samples[0][0])
                
                # cv2.imwrite('inference_resuts/raw_img_{}_128.png'.format(num), test_img[0][0] * 255)
                # cv2.imwrite('inference_resuts/raw_mask_{}_128.png'.format(num), test_mask[0][0] * 255)
                t1 = time.time()
                print(num, count, ' inference time: ', t1 - t0)
                count += 1

    def plot_samples(self, samples):
        """Plot the batch of samples.

        Args:
            samples: a mini_batch of samples to plot
        """
        plt.figure(figsize=(6, 6))
        grid = make_grid(samples)
        np_grid = grid.numpy().transpose((1, 2, 0))
        plt.imshow(np_grid*np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5]))
        plt.axis("off")
        
    def plot_losses(self):
        """Plot the training and the validation losses."""
        
        plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        