class Upsampling(Module):
    """NNUpsampling + Convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1,padding = 0, stride = 1,scale_factor = 1):
        """Initialize parameters."""
        self.type = 'ConvTranspose2d'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale_factor
        self.stride = stride
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.conv = Conv2d(self.in_channels,self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        [(self.weight, self.w_grad), (self.bias, self.bias_grad)] = self.conv.param()

    def forward(self,input):
        """NNUpsampling forward"""
        self.input = input 
        a,b,c,d = self.input.shape
        tensor = self.input.reshape(a*b*c,d)
        m,n = tensor.shape
        row_idx = ((arange(1, 1 + m*self.scale)/self.scale).ceil() - 1).type(torch.long)
        col_idx = ((arange(1, 1 + n*self.scale)/self.scale).ceil() - 1).type(torch.long)
        res = tensor[row_idx,:][:,col_idx]
        self.nnupsampling = res.reshape(a,b,c*self.scale,-1)

        """Convolution forward"""
        output = self.conv.forward(self.nnupsampling)
        return output 
    
    def backward(self,gradwrtoutput):
        """Convolution backward"""
        self.gradwrtconv = self.conv.backward(gradwrtoutput)
        [(self.weight, self.w_grad), (self.bias, self.bias_grad)] = self.conv.param()
        
        """NNupsampling backward"""
        conv = Conv2d(1,1,scale,scale,bias=False)
        conv.weight = torch.ones((1,1,scale,scale))
        a,b,c,d = self.gradwrtconv.shape
        gradwrtinput = conv.forward(self.gradwrtconv.reshape(1,1,a*b*c,d))
        gradwrtinput = gradwrtinput.reshape(self.input.shape)
        return gradwrtinput

    def param(self):
        return [(self.weight, self.w_grad), (self.bias, self.bias_grad)]