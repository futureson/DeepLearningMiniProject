import torch
from torch import arange

def stride_pad(tensor, stride):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N = 1+2*(stride-1)
    a,b,m,n = tensor.shape
    out_c = torch.zeros((a,b,m,n*stride-(stride-1))).to(device)
    out_c[:,:,:,arange(n)*stride] = tensor
    out_r = torch.zeros((a,b,m*stride-(stride-1),n*stride-(stride-1))).to(device)
    out_r[:,:,arange(m)*stride,:] = out_c
    return out_r

def arround_pad(tensor,h,w):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    a,b,m,n = tensor.shape
    out = torch.zeros((a,b,m+2*h,n+2*w)).to(device)
    out[:,:,h:h+m,w:w+n] = tensor
    return out

def inverse_stride(tensor,stride):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    p = arange(1,tensor.shape[2]+1,stride)-1
    q = arange(1,tensor.shape[3]+1,stride)-1
    row = tensor[:,:,p,:]
    res = row[:,:,:,q]
    return res

def inverse_arround(tensor,h,w):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    a,b,m,n = tensor.shape
    out = torch.zeros((a,b,m-2*h,n-2*w)).to(device)
    out = tensor[:,:,h:m-h,w:n-w]
    return out

def output_pad(tensor,h,w):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    a,b,m,n = tensor.shape
    out = torch.zeros((a,b,m+h,n+w)).to(device)
    out[:,:,0:m,0:n] = tensor
    return out

def inverse_out(tensor,h,w):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    a,b,m,n = tensor.shape
    out = torch.zeros((a,b,m-h,n-w)).to(device)
    out = tensor[:,:,0:m-h,0:n-w]
    return out

def psnr(x,y, max_range=1.0):
    """Compute peak signal-to-noise ratio."""
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

