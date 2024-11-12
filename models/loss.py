import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

LAMBDA_EDGE = 0.05 # 0.005 for fft-mse loss or 0.05 for charbonnier_edge
LAMBDA_FFT = 0.005
LAMBDA_LPIPS = 0.5
print(f'Lambda_edge: {LAMBDA_EDGE}')
print(f'Lambda_fft: {LAMBDA_FFT}')
print(f'Lambda_lpips: {LAMBDA_LPIPS}')

def mse_loss(output, target):
    return F.mse_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

##############
### Losses on image
##############

def fft_loss(output_noise, target_noise, output_img, target_img):
    img_fft = torch.fft.rfft2(output_img)
    img_fft = torch.fft.fftshift(img_fft)

    target_fft = torch.fft.rfft2(target_img)
    target_fft = torch.fft.fftshift(target_fft)

    loss_fft_img = l1_loss(img_fft, target_fft)
    loss_mse_noise = mse_loss(output_noise, target_noise)

    return loss_mse_noise + LAMBDA_FFT * loss_fft_img

def charbonnier_edge_loss(output_noise, target_noise, output_img, target_img):
    epsilon = 1e-6
    diff = output_noise - target_noise

    ch_loss = torch.mean(torch.sqrt(diff ** 2 + epsilon ** 2))

    laplacian_kernel = torch.tensor([[[[0, 1, 0], 
                                        [1, -4, 1], 
                                        [0, 1, 0]]]], dtype=torch.float32)
    laplacian_kernel = laplacian_kernel.to(output_img.device)

    output_laplacian = F.conv2d(output_img, laplacian_kernel, padding=1)
    target_laplacian = F.conv2d(target_img, laplacian_kernel, padding=1)

    ed_loss = F.mse_loss(output_laplacian, target_laplacian)

    return ch_loss + LAMBDA_EDGE * ed_loss

def lpips_loss(output_noise, target_noise, output_img, target_img):
    loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(target_img.device)
    loss_lpips = loss_fn(output_img.repeat(1, 3, 1, 1), target_img.repeat(1, 3, 1, 1)) # lpips takes RGB as input
    loss_mse_noise = mse_loss(output_noise, target_noise)

    return loss_mse_noise + LAMBDA_LPIPS*loss_lpips
    
