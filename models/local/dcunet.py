import torch
import torch.nn as nn


class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        # self.embd_dim = embd_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.stride = stride
        
        # self.real_conv = nn.Conv2d(in_channels=self.in_channels+1, 
        #                            out_channels=self.out_channels, 
        #                            kernel_size=self.kernel_size, 
        #                            padding=self.padding, 
        #                            stride=self.stride)
        
        # self.im_conv = nn.Conv2d(in_channels=self.in_channels+1, 
        #                          out_channels=self.out_channels, 
        #                          kernel_size=self.kernel_size, 
        #                          padding=self.padding, 
        #                          stride=self.stride)

        self.wav_conv = nn.Conv2d(in_channels=self.in_channels+1, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        # self.emb_conv = nn.Conv2d(in_channels=self.in_channels, 
        #                          out_channels=self.out_channels, 
        #                          kernel_size=self.kernel_size, 
        #                          padding=self.padding, 
        #                          stride=self.stride)

        # self.linear = nn.Linear(emb, self.out_channels)
        
        # Glorot initialization.q
        # nn.init.xavier_uniform_(self.real_conv.weight)
        # nn.init.xavier_uniform_(self.im_conv.weight)
        nn.init.xavier_uniform_(self.wav_conv.weight)
        # nn.init.xavier_uniform_(self.emb_conv.weight)
        
        
    def forward(self, x):
        # x_real = x[..., 0]
        # x_im = x[..., 1]

        # emb_real = emb[..., 0]
        # emb_im = emb[..., 1]

        # # # print(x_real.shape, aux.shape)

        # c_real = self.real_conv(torch.cat([x_real,emb_real],dim=1)) - self.im_conv(torch.cat([x_im,emb_im],dim=1))
        # c_im = self.im_conv(torch.cat([x_real,emb_real],dim=1)) + self.real_conv(torch.cat([x_im,emb_im],dim=1))

        # emb_real = self.emb_real_conv(emb_real)
        # emb_im = self.emb_im_conv(emb_im)

        # # c_real = self.real_conv(x_real) - self.im_conv(x_im)
        # # c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        # output = torch.stack([c_real, c_im], dim=-1)
        # emb = torch.stack([emb_real, emb_im], dim=-1)
        
        # x = torch.cat([x, emb], dim=1)
        # linear
        output = self.wav_conv(x)
        # emb = self.emb_conv(emb)
        # out_put = self.fusion_conv(torch.cat([output, emb], dim=1))
        # emb = self.emb_conv(emb)

        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        # self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
        #                                     out_channels=self.out_channels, 
        #                                     kernel_size=self.kernel_size, 
        #                                     output_padding=self.output_padding,
        #                                     padding=self.padding,
        #                                     stride=self.stride)
        
        # self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
        #                                     out_channels=self.out_channels, 
        #                                     kernel_size=self.kernel_size, 
        #                                     output_padding=self.output_padding, 
        #                                     padding=self.padding,
        #                                     stride=self.stride)
        
        self.wav_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        # self.emb_conv = nn.Conv2d(in_channels=1, 
        #                          out_channels=1, 
        #                          kernel_size=self.kernel_size, 
        #                          padding=self.padding, 
        #                          stride=self.stride)
        
        
        # Glorot initialization.
        # nn.init.xavier_uniform_(self.real_convt.weight)
        # nn.init.xavier_uniform_(self.im_convt.weight)
        nn.init.xavier_uniform_(self.wav_convt.weight)
        
        
    def forward(self, x):
        # x_real = x[..., 0]
        # x_im = x[..., 1]
        
        # ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        # ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        # output = torch.stack([ct_real, ct_im], dim=-1)

        output = self.wav_convt(x)
        return output

class Encoder_CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
        #                               affine=self.affine, track_running_stats=self.track_running_stats)
        # self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
        #                             affine=self.affine, track_running_stats=self.track_running_stats)
        self.wav_norm = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats)
        # self.emb_norm = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
        #                             affine=self.affine, track_running_stats=self.track_running_stats)  
        # self.emb_real_norm = nn.BatchNorm2d(num_features=1, eps=self.eps, momentum=self.momentum,
        #                             affine=self.affine, track_running_stats=self.track_running_stats)
        # self.emb_im_norm = nn.BatchNorm2d(num_features=1, eps=self.eps, momentum=self.momentum,
        #                             affine=self.affine, track_running_stats=self.track_running_stats)
        
    def forward(self, x):
        # print(x.shape)
        # x_real = x[..., 0]
        # x_im = x[..., 1]

        # emb_real = emb[..., 0]
        # emb_im = emb[..., 1]
        
        # n_real = self.real_b(x_real)
        # n_im = self.im_b(x_im) 
        
        # n_emb_real = self.emb_real_norm(emb_real)
        # n_emb_im = self.emb_im_norm(emb_im)
        
        # output = torch.stack([n_real, n_im], dim=-1)
        # emb = torch.stack([n_emb_real, n_emb_im], dim=-1)
        # print(emb.shape)
        output = self.wav_norm(x)
        # emb = self.emb_norm(emb)
        return output

class Decoder_CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
        #                               affine=self.affine, track_running_stats=self.track_running_stats)
        # self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
        #                             affine=self.affine, track_running_stats=self.track_running_stats)
        self.wav_norm = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        # x_real = x[..., 0]
        # x_im = x[..., 1]
        
        # n_real = self.real_b(x_real)
        # n_im = self.im_b(x_im)  
        
        # output = torch.stack([n_real, n_im], dim=-1)
        output = self.wav_norm(x)
        return output

class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = Encoder_CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        # aux = aux.repeat(1,1,x.shape[2],x.shape[3])
        # print(x.shape, aux.shape)
        conved = self.cconv(x)
        # print(conved.shape, emb.shape)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        # emb = self.leaky_relu(emb)
        
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = Decoder_CBatchNorm2d(num_features=self.out_channels) 
        
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output