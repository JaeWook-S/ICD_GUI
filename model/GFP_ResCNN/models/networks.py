import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from model.GFP_ResCNN.models import residual_transformers
##############################################################################
import torchvision.models as models
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1) # epoch이  niter 이상하면 학습률 감소
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG,vit_name,img_size,pre_trained_path, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[],pre_trained_trans=True,pre_trained_resnet=0):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available()) # assert 괄호 안이 True가 아니면 AssertError가 발생
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'res_cnn':
        netG = residual_transformers.Res_CNN(residual_transformers.CONFIGS[vit_name], input_dim= input_nc, img_size=img_size, output_dim=1, vis=False)
    elif which_model_netG == 'resvit':
        print(vit_name)
        netG = residual_transformers.ResViT(residual_transformers.CONFIGS[vit_name],input_dim = input_nc,img_size=img_size, output_dim=1, vis=False)
        config_vit = residual_transformers.CONFIGS[vit_name]
        if pre_trained_resnet:
            pre_trained_model = residual_transformers.Res_CNN(residual_transformers.CONFIGS[vit_name], input_dim= input_nc, img_size=img_size, output_dim=1, vis=False)
            save_path = pre_trained_path
            print("pre_trained_path: ",save_path)
            pre_trained_model.load_state_dict(torch.load(save_path))

            pretrained_dict = pre_trained_model.state_dict()
            model_dict = netG.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            netG.load_state_dict(model_dict)

            print("Residual CNN loaded")

        if pre_trained_trans:
            print(config_vit.pretrained_path)
            netG.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, which_model_netD,vit_name,img_size,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic': # 현재 이거 사용
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel() # numel : 파라미터 개수 
    print(net)
    print('Total number of parameters: %d' % num_params)
    

## 내가 수정한 코드 - SSIM loss 사용해보자
def ssim_loss(fake_B, real_B, window_size=11, C1=0.01**2, C2=0.03**2):
    def create_window(window_size, channel):
        _1D_window = torch.hann_window(window_size, periodic=False).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.T
        _2D_window /= _2D_window.sum()
        window = _2D_window.expand(channel, 1, window_size, window_size)
        return window.to(fake_B.device)

    channel = fake_B.shape[1]
    window = create_window(window_size, channel)

    mu1 = F.conv2d(fake_B, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(real_B, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(fake_B * fake_B, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(real_B * real_B, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(fake_B * real_B, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()


##############################################################################
# Classes
##############################################################################

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # Max pooling along channel axis
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average pooling along channel axis
        
        attn = torch.cat([max_pool, avg_pool], dim=1)  # Concatenate along channel axis
        attn = self.conv(attn)  # Apply convolution
        attn = self.sigmoid(attn)  # Apply sigmoid activation
        
        return x * attn  # Element-wise multiplication with input tensor
    
    

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module): # 생성자가 판별기를 속이도록 학습 진행하여야 함
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel())) # real_label_var이 없거나 크기가 다르면 생성
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label) # input size만큼 real_label 1 값으로 채움
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real): # target_is_real : 이게 진짜 이미지인가? // generator는 진짜라고 속여야되므로 True로 설정함 // discriminator는 잘 판단해야하므로 진짜 이미지가 True임
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

## 내가 수정한 코드 - VGG loss 사용해보자

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        vgg = models.vgg19(pretrained=True).features
        #conv3_3층까지만 사용
        self.feature_extractor = nn.Sequential(*[vgg[i] for i in range(16)])
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.device = device
        self.feature_extractor.to(self.device)
    
    def forward(self, x):
        return self.feature_extractor(x.to(self.device))

class VGGLoss(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.vgg = VGGFeatureExtractor(device=device)
        self.criterion = nn.L1Loss()
    
    def forward(self, fake_B, real_B):
        # 둘 다 green채널만 쓰기 때문에 1채널임 -> 3채널로 변경 후 적용
        fake_B_3channel = fake_B.repeat(1, 3, 1, 1)
        real_B_3channel = real_B.repeat(1, 3, 1, 1)
        
        fake_B_features = self.vgg(fake_B_3channel)
        real_B_features = self.vgg(real_B_3channel)
        
        loss = self.criterion(fake_B_features, real_B_features)
        
        return loss



class Encoder_Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',down_samp=1,gated_fusion=0):
        super(Encoder_Decoder, self).__init__()        
        self.output_nc = output_nc      
        self.encoders=2
        latent_size=16
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #Encoders
        for ii in range(2):
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                     norm_layer(ngf), nn.ReLU(True)]   
            n_downsampling = 2 
            
            ### downsample
            for i in range(n_downsampling):
                mult = 2**i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2), nn.ReLU(True)]
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            #model += [nn.ReflectionPad2d(1)]              
            model += [nn.Conv2d(ngf * mult, latent_size, kernel_size=3, padding=1), 
                     norm_layer(latent_size), nn.ReLU(True)]   
            setattr(self, 'model_enc_'+str(ii), nn.Sequential(*model))
        #Decoder
        #model += [nn.ReflectionPad2d(3)] 
        model = [nn.Conv2d(latent_size*2, 256, kernel_size=3, padding=1), 
                 norm_layer(256), nn.ReLU(True)]  
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1,bias=use_bias),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        setattr(self, 'model_dec', nn.Sequential(*model))

            
            
    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        encoded=[]
        for ii in range(self.encoders):
            encoded.append( getattr(self, 'model_enc_'+str(ii))(input[:,ii,:,:]))
        decoded=self.model_dec(torch.cat((encoded[0],encoded[1]),1))
        return decoded
#        else:
#            return self.model(input)







# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect',down_samp=1):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.down_samp=down_samp
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
############################################################################################
#Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'model_1', nn.Sequential(*model)) 
############################################################################################
#Layer2-Encoder2
        n_downsampling = 2
        model = []
        i=0
        mult = 2**i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        setattr(self, 'model_2', nn.Sequential(*model))
############################################################################################
#Layer3-Encoder3
        model = []
        i=1
        mult = 2**i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                        stride=2, padding=1, bias=use_bias),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]
        setattr(self, 'model_3', nn.Sequential(*model))
############################################################################################
#Layer4-Residual1
        mult = 2**n_downsampling
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_4', nn.Sequential(*model))
############################################################################################
#Layer5-Residual2
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_5', nn.Sequential(*model))
############################################################################################
#Layer6-Residual3
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_6', nn.Sequential(*model))
############################################################################################
#Layer7-Residual4
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_7', nn.Sequential(*model))
############################################################################################
#Layer8-Residual5
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_8', nn.Sequential(*model))
############################################################################################
#Layer9-Residual6
        model = []
        use_dropout = norm_layer == nn.InstanceNorm2d
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_9', nn.Sequential(*model))
############################################################################################
#Layer10-Residual7
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_10', nn.Sequential(*model))
############################################################################################
#Layer11-Residual8
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_11', nn.Sequential(*model))
############################################################################################
#Layer12-Residual9
        model = []
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        setattr(self, 'model_12', nn.Sequential(*model))
############################################################################################
#Layer13-Decoder1
        i = 0
        mult = 2**(n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                                    norm_layer(int(ngf * mult / 2)),
                                    nn.ReLU(True)]
        setattr(self, 'model_13', nn.Sequential(*model))
############################################################################################
#Layer14-Decoder2
        i = 1
        mult = 2**(n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                                    norm_layer(int(ngf * mult / 2)),
                                    nn.ReLU(True)]
        setattr(self, 'model_14', nn.Sequential(*model))
############################################################################################
#Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'model_15', nn.Sequential(*model))
############################################################################################


    def forward(self, input):
        x1 = self.model_1(input)
        x2 = self.model_2(x1)
        x3 = self.model_3(x2)
        x4 = self.model_4(x3)
        x5 = self.model_5(x4)
        x6 = self.model_6(x5)
        x7 = self.model_7(x6)
        x8 = self.model_8(x7)
        x9 = self.model_9(x8)
        x10 = self.model_10(x9)
        x11 = self.model_11(x10)
        x12 = self.model_12(x11)
        x13 = self.model_13(x12)
        x14 = self.model_14(x13)
        x15 = self.model_15(x14)
        return x15
        

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial: # nn.BatchNorm2d는 자체적으로 bias수행하므로 use_bias=False임
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4 # 커널 크기
        padw = 1 # 패딩 크기
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True) # relu에서 음수 값을 0이 아닌 일정비율(기울기:0.2)인 0.2 * x 로 남기는 함수 // inplace=True -> 메모리 사용량 줄이기 위해 입력 텐서 자체를 수정
        ] # GAN discriminator는 음수 값도 활용해야하므로 leaky가 적합

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8) # 채널 수 증가 -> 최대 8배까지
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # 최종적으로 이미지가 1인지 0인지 판별하기 위해 channel 1로 

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor): # isinstance : input이 torch.cuda.FloatTensor인지 확인
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids) # 멀티 GPU 연산 사용
        else:
            print(self.model(input).size())
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)
