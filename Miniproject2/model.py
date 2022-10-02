import torch
import pickle
import math
import os
from torch import empty
from torch.nn.functional import fold, unfold
from collections import OrderedDict
from .others.helpers import *
from pathlib import Path


# autograd globally off
torch.set_grad_enabled(False)


class Model():
    def __init__(self,in_channels = 3,out_channels = 3,lr = 0.01,momentum = 0.9,batch_size = 16) -> None:
        """
        Initialize all model + criterion + optimizer
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        
        self.model =  Sequential(
                    Conv2d(self.in_channels, 48, 3, stride=2, padding=1),
                    ReLU(),
                    Conv2d(48, 48, 3, stride=2, padding=1),
                    ReLU(),
                    Upsampling(48, 48, 3, padding=1, stride=2, output_padding=1),
                    ReLU(),
                    Upsampling(48, self.out_channels, 3, padding=1, stride=2, output_padding=1),
                    Sigmoid()
                    )
        
        self.criterion = MSE()
        self.optim = SGD(params=self.model.params(), lr=self.lr, momentum=self.momentum)



    def load_pretrained_model(self) -> None:
        """
        load the best pretrained model from checkpoint files.
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'rb') as files:
            params = pickle.load(files)
        self.model.load_para(params)
        self.model.load()

    def train(self,train_input,train_target,num_epochs, normalize=True) -> None:
        """
        train a new model
        """
        # Performs Tensor dtype and/or device conversion
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)

        if normalize:
            train_input = train_input/255.0
            train_target = train_target/255.0
        
        for epoch in range(num_epochs):
            for train, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                # forward pass
                output = self.model.forward(train)
                loss = self.criterion.forward(output, target)
                dl_dy = self.criterion.backward()
                
                # backward pass + optimization
                self.optim.zero_grad()
                self.model.backward(dl_dy)
                self.optim.step()   
   
    def predict(self,test_input) -> torch.Tensor:
        """
        Testing using unseen images
        """
        test_input = test_input.float() / 255.0  
        test_input = test_input.to(self.device)
        
        y_hat = self.model.forward(test_input)
        y_hat = y_hat.clamp(0,1)
        return y_hat * 255.0
    
    def save_model(self, path) -> None:
        """
        path: the path name (string) to store model parameters
        Save model
        """
        with open('bestmodel.pth', 'wb') as files:
            pickle.dump(self.model.param(), files)


""" base class of all own modules"""
class Module(object):
    def __init__(self):

        raise NotImplementedError
        
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []
    
    def zero_grad(self):
        raise []
    
    
class ReLU(Module):
    def __init__(self):
        self.type = 'ReLU'
        self.input = None
        self.output = None
        self.gradwrtinput = None
        
    def forward(self, input):
        self.input = input
        self.output = input * (input > 0)
        return self.output
    
    def backward(self, gradwrtoutput):
        self.gradwrtinput = gradwrtoutput * (self.input > 0)
        return self.gradwrtinput
    
    def param(self):
        return []
    
    def zero_grad(self):
        pass

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.type = 'LeakyReLU'
        self.input = None
        self.output = None
        self.gradwrtinput = None
        self.negative_slope = negative_slope
        
    def forward(self, input):
        self.input = input
        self.output = input * (input >= 0) + self.negative_slope * input * (input < 0)
        return self.output
    
    def backward(self, gradwrtoutput):
        self.gradwrtinput = gradwrtoutput * (self.input >= 0) + gradwrtoutput * self.negative_slope * (self.input < 0)
        return self.gradwrtinput
    
    def param(self):
        return []
    
    def zero_grad(self):
        pass
    
class MSE(Module):
    """
    Module to measure the mean square error (MSE) 
    """
    def __init__(self):
        self.type = 'MSE'
        self.error = None
        self.loss = None
        self.gradwrtinput = None
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Returns the MSE Loss between pred and target
        Args:
            pred (Tensor): predicted values
            target (Tensor): ground truth values
        Returns:
            Tensor: MSE Loss
        """
        self.error = (pred - target)
        self.loss = (self.error ** 2).mean()
        return self.loss
    
    def backward(self):
        """
        Returns:
            Tensor: the gradient of MSE Loss
        """
        self.gradwrtinput = 2 * self.error / self.error.nelement()
        return self.gradwrtinput
    
    def param(self):
        return []
    
    def zero_grad(self):
        pass
    

class Sigmoid(Module):
    def __init__(self):
        self.type = 'Sigmoid'
        self.input = None
        self.output = None
        self.gradwrtinput = None
        
    def forward(self, input):
        self.input = input
        self.output = 1.0 / (1 + (-1 * input).exp()) 
        return self.output
    
    def backward(self, gradwrtoutput):
        self.gradwrtinput = gradwrtoutput * (self.output * (1 - self.output))
        return self.gradwrtinput
    
    def param(self):
        return []
    
    def zero_grad(self):
        pass
    
class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input
    planes."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, weightsinit: str = "uniform"):
        """Initialize parameters."""
        self.type = 'Conv2d'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
            
        self.stride = stride 
        self.use_bias = bias
        
        self.weightsinit = weightsinit
        
        self.weight = torch.empty([out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]])
        self.bias = torch.empty([out_channels])
        
        self.w_grad = torch.empty([out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]]).zero_()
        self.bias_grad = torch.empty([out_channels]).zero_()
        
        self.fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        
        self.init_weight()
        
    def init_weight(self):
        """
        Initialises the weight and bias parameters of the layer depending on
        the weightinit parameter. 
        If "weightsinit" is
            1. uniform (by default), the weights are initiliased using uniform distribution (same as Pytorch).
            2. xavier, the weights are initiliased using Xavier Initialisation.
            3. kaiming, the weights are initiliased using Kaiming Initialisation when using ReLU layer
        """
        
        if self.weightsinit == "uniform":
            K = math.sqrt(1/(self.fan_in))
            # Initialize kernel weights + zero out gradients
            self.weight.uniform_(-K, K)
            # Initialize kernel bias + zero out gradients
            self.bias.uniform_(-K, K)
            
        elif self.weightsinit == "xavier":
            self.weight.normal_(0, math.sqrt(2/(self.fan_in + self.fan_out)))
            self.bias.zero_()
            
        elif self.weightsinit == "kaiming":
            self.weight.normal_(0, math.sqrt(2/(self.fan_in)))
            self.bias.zero_()
            

    
    def forward(self, input):
        """
        Args:
            input (Tensor): size = (batch size, in_channels, H, W)        
        Returns:
            actual (Tensor): convolution resutls, size = (batch size, out_channels, .., ..)
        """
        self.input = input
        self.weight = self.weight.to(self.device)
        self.w_grad = self.w_grad.to(self.device)
        self.bias = self.bias.to(self.device)
        self.bias_grad = self.bias_grad.to(self.device)
        
        input_unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        wxb = self.weight.view(self.out_channels, -1) @ input_unfolded
        if self.use_bias:
            wxb += self.bias.view(1, -1, 1)
        
        out_H = math.floor((self.input.shape[2]-self.kernel_size[0]+2*self.padding[0])/self.stride)+1

        return wxb.view(self.input.shape[0], self.out_channels, out_H, -1)
    

    
    def backward(self, gradwrtoutput):
        """
        Args:
            gradwrtoutput (Tensor): grad with respect to output, size = (batch size, out_channels, .., ..)
        Returns:
            gradwrtinput (Tensor): grad with respect to input, size = (batch size, C, H, W)
        """
        self.weight = self.weight.to(self.device)
        self.w_grad = self.w_grad.to(self.device)
        self.bias = self.bias.to(self.device)
        self.bias_grad = self.bias_grad.to(self.device)
        
        # size issue
        H = self.input.shape[2] + 2 * self.padding[0] - self.kernel_size[0] + 1 
        W = self.input.shape[3] + 2 * self.padding[1] - self.kernel_size[1] + 1 
        
        # ∂L/∂w is nothing but the convolution between Input X and Loss Gradient from the next layer ∂L/∂O
      
        if(self.stride>1):
            zero_out = torch.zeros(gradwrtoutput.shape[0], self.out_channels, H, W)
            zero_out[:,:,::self.stride, ::self.stride] = gradwrtoutput
            gradout_unfold = zero_out.permute(1, 0, 2, 3).reshape(self.out_channels, -1)
        else:
            gradout_unfold = gradwrtoutput.permute(1, 0, 2, 3).reshape(self.out_channels, -1)
        gradout_unfold = gradout_unfold.to(self.device)
        input_unfolded = unfold(self.input.permute(1, 0, 2, 3), kernel_size=(H, W), padding=self.padding)

        self.w_grad = (gradout_unfold @ input_unfolded).permute(1, 0, 2).view(self.out_channels, self.in_channels, self.kernel_size[0], -1)

        if self.use_bias:
            # ∂L/∂b is sum of ∂L/∂O for each kernel
            self.bias_grad = gradwrtoutput.sum(axis = (0, 2, 3)) 
        
        # ∂L/∂X can be represented as ‘full’ convolution between a 180-degree rotated Filter F and loss gradient ∂L/∂O
        # rotate + permute same channel kernel + flat
        w_unfold = self.weight.flip([2, 3]).permute(1, 0, 2, 3).reshape(self.weight.shape[1],-1)
        
        padding_0 = self.kernel_size[0]-1
        padding_1 = self.kernel_size[1]-1
        
        # insert 0s internally
        if(self.stride>1):
            zero_out = torch.zeros(gradwrtoutput.shape[0], self.out_channels, H, W)
            zero_out[:,:,::self.stride, ::self.stride] = gradwrtoutput
            gradout_unfold = unfold(zero_out, kernel_size=self.kernel_size, padding=(padding_0,padding_1))
        else:
            gradout_unfold = unfold(gradwrtoutput, kernel_size=self.kernel_size, padding=(padding_0,padding_1))
        gradout_unfold = gradout_unfold.to(self.device)
        gradwrtinput_unfold = w_unfold @ gradout_unfold
        
        if self.padding != 0:
            gradwrtinput = gradwrtinput_unfold.view((self.input.shape[0], self.input.shape[1], self.input.shape[2]+2*self.padding[0], self.input.shape[3]+2*self.padding[1]))[:,:,1:-1,1:-1]
        else:
            gradwrtinput = gradwrtinput_unfold.view(self.input.shape)

        return gradwrtinput
        
    def param(self):
        """Return parameters."""
        return [(self.weight, self.w_grad), (self.bias, self.bias_grad)]
    
    def zero_grad(self):
        self.w_grad.fill_(0.)
        self.bias_grad.fill_(0.)
    
class SGD(Module):
    def __init__(self, params, lr, momentum=0):
        """
        params = a list of layers
        lr = learning rate
        momentum = optionally with momentum
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.lr = lr
        if self.lr <= 0.0:
            raise ValueError(
                "Learning rate {} should be greater than zero".format(self.lr))
        
        # momentum
        self.momentum = momentum
        self.params = params
        self.set_velocity()
        

    def set_velocity(self):
        """
        Initialize the velocity for weights and bias
        """
        self.w_velocity = []
        self.bias_velocity = []
        
        for layer in self.params:
            if layer.type in ['Conv2d', 'ConvTranspose2d']:
                self.w_velocity.append(0.*layer.w_grad)
                self.bias_velocity.append(0.*layer.bias_grad)
            else:
                self.w_velocity.append(torch.tensor([0]))
                self.bias_velocity.append(torch.tensor([0]))
                
            
    def zero_grad(self):
        """
        Zero out all gradients
        """
        for layer in self.params:
            if layer.type in ['Conv2d', 'ConvTranspose2d']:
                layer.w_grad.fill_(0.)
                layer.bias_grad.fill_(0.)
            
    def step(self):
        """
        ref: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        return updated weight and bias (tensor)
        """        
        
        for idx, layer in enumerate(self.params):
            if layer.type in ['Conv2d', 'ConvTranspose2d']:
                self.w_velocity[idx] = self.w_velocity[idx].to(self.device) 
                self.w_velocity[idx] = self.momentum * self.w_velocity[idx] + layer.w_grad
                layer.weight = layer.weight - self.lr * self.w_velocity[idx]
                
                if layer.use_bias:
                    # check if layer uses bias
                    self.bias_velocity[idx] = self.bias_velocity[idx].to(self.device)
                    self.bias_velocity[idx] = self.momentum * self.bias_velocity[idx] + layer.bias_grad
                    layer.bias = layer.bias - self.lr * self.bias_velocity[idx]

                
class Upsampling(Module):
        #"""Applies a 2D transpose convolution over an input signal composed of several input planes."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1,padding = 0, stride = 1,output_padding = 0, bias=True, weightsinit: str = "uniform"):
        """Initialize parameters."""
        self.type = 'ConvTranspose2d'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding            
            
        self.stride = stride 
        self.use_bias = bias
        
        self.weightsinit = weightsinit
        
        self.weight = torch.empty([in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]])
        self.bias = torch.empty([out_channels])
        
        self.w_grad = torch.empty([in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]]).zero_()
        self.bias_grad = torch.empty([out_channels]).zero_()
        
        self.fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        
        self.init_weight()
        
    def init_weight(self):
        """
        Initialises the weight and bias parameters of the layer depending on
        the weightinit parameter. 
        If "weightsinit" is
            1. uniform (by default), the weights are initiliased using uniform distribution (same as Pytorch).
            2. xavier, the weights are initiliased using Xavier Initialisation.
            3. kaiming, the weights are initiliased using Kaiming Initialisation when using ReLU layer
        """
        
        if self.weightsinit == "uniform":
            K = math.sqrt(1/(self.fan_in))
            # Initialize kernel weights + zero out gradients
            self.weight.uniform_(-K, K)
            # Initialize kernel bias + zero out gradients
            self.bias.uniform_(-K, K)
            
        elif self.weightsinit == "xavier":
            self.weight.normal_(0, math.sqrt(2/(self.fan_in + self.fan_out)))
            self.bias.zero_()
            
        elif self.weightsinit == "kaiming":
            self.weight.normal_(0, math.sqrt(2/(self.fan_in)))
            self.bias.zero_()

    def forward(self, input):
        self.weight = self.weight.to(self.device)
        self.w_grad = self.w_grad.to(self.device)
        self.bias = self.bias.to(self.device)
        self.bias_grad = self.bias_grad.to(self.device)
        
        self.input = output_pad(arround_pad(stride_pad(input,self.stride),(self.kernel_size[0]-1),(self.kernel_size[1]-1)),self.output_padding[0],self.output_padding[1])
        input_unfolded = unfold(self.input, kernel_size=self.kernel_size)
        weight = self.weight.permute(1,0,2,3).flip(dims = [3,2]) #Temporary variable
        wxb = weight.reshape(self.out_channels, -1) @ input_unfolded
        if self.use_bias:
            wxb += self.bias.view(1, -1, 1)
        ## self.input.shape[0] = batch size
        res = wxb.view(self.input.shape[0], self.out_channels, self.input.shape[2]-self.kernel_size[0]+1, -1)
        actual = inverse_arround(res,self.padding[0],self.padding[1])
        return actual

    def backward(self, gradwrtoutput):
        gradwrtoutput = arround_pad(gradwrtoutput,self.padding[0],self.padding[1])
        gradwrtoutput = gradwrtoutput.to(self.device)
        if self.use_bias:
            self.bias_grad = gradwrtoutput.sum(axis = (0, 2, 3)) 

        dwxb = gradwrtoutput.view(gradwrtoutput.shape[0],gradwrtoutput.shape[1],-1)
        input_unfolded = unfold(self.input, kernel_size=self.kernel_size)
        dw1 = dwxb @ input_unfolded.permute(0,2,1)
        dw1 = dw1.sum(0)
        dw2 = dw1.view(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
        self.w_grad = dw2.permute(1,0,2,3).flip(dims = [3,2])
        

        dx1 = self.weight.permute(1,0,2,3).flip(dims = [3,2]).reshape(self.out_channels,-1).T @ dwxb 
        dx2 = fold(dx1,self.input.shape[-2:],self.kernel_size)
        gradwrtinput = inverse_stride(inverse_arround(inverse_out(dx2,self.padding[0],self.padding[1]),self.kernel_size[0]-1,self.kernel_size[1]-1),self.stride)
        return gradwrtinput

    def param(self):
        return [(self.weight, self.w_grad), (self.bias, self.bias_grad)]   
    
    def zero_grad(self):
        self.w_grad.fill_(0.)
        self.bias_grad.fill_(0.)
                
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=-1, padding=0):
        self.type = 'MaxPool2d'
        self.input = None
        self.gradwrtinput = None
        self.kernel_size = kernel_size
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.padding = padding
        
    def forward(self, input):
        self.input = input
        H = math.floor((input.shape[2] + 2 * self.padding - self.kernel_size)/self.stride) + 1 
        self.input_unfold = unfold(input, kernel_size=self.kernel_size, stride=self.stride, \
                                   padding=self.padding).reshape(input.shape[0], input.shape[1],\
                                                                 self.kernel_size*self.kernel_size, -1)
 
        self.values, self.indices = torch.max(self.input_unfold, 2)
        
        pool_output = self.values.reshape(input.shape[0], input.shape[1], H, -1)

        return pool_output
    
    def backward(self, gradwrtoutput):
        onehotSplit = torch.empty(self.input_unfold.shape).zero_().scatter_(dim=2,\
                            index=self.indices.reshape(self.input_unfold.shape[0],self.input_unfold.shape[1],-1,self.input_unfold.shape[3]).long(), value=1)

        result = gradwrtoutput.reshape(gradwrtoutput.shape[0],gradwrtoutput.shape[1],1,-1) * onehotSplit

        self.gradwrtinput = fold(result.reshape(result.shape[0], -1, result.shape[3]), self.input.shape[-2:], self.kernel_size, stride=self.stride)

        return self.gradwrtinput
    
    def param(self):
        return []
    
    def zero_grad(self):
        pass         
         
        
class Sequential(Module):
    """Arbitrary configuration of modules together."""
    def __init__(self, *args):
        """
        Args:
            *args (list[layer]): a series of layers      
        """
        self.layers = [layer for layer in args]
        
    def forward(self, input):
        """
        Forward propagating the input to produce the final output
        Returns:
            input (tensor): output of final layer      
        """
        self.input = input
        output = input   # in case of no layers
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output

        return self.output
    
    def backward(self, gradwrtoutput):
        """
        Backward propagating the gradient from the last layer to the first
        Returns:
            grad (tensor): the gradient of the first layer  
        """
        self.gradwrtoutput = gradwrtoutput
        for layer in reversed(self.layers):
            gradwrtoutput = layer.backward(gradwrtoutput)
        return gradwrtoutput
    
    def zero_grad(self):
        """
        set all gradient to zeros
        """
        for layer in self.layers:
            layer.zero_grad()
            
    def params(self):
        return self.layers
    
    def load_para(self,input_params):
        self.input_params  = input_params
        return self.input_params
    
    def load(self):
        for idx, layer in enumerate(self.layers):
             if layer.type in ['Conv2d', 'ConvTranspose2d']:
                layer.weight = self.input_params[idx][0]
                layer.w_grad = self.input_params[idx][1]
                layer.bias = self.input_params[idx+1][0]
                layer.bias_grad = self.input_params[idx+1][1]
    
