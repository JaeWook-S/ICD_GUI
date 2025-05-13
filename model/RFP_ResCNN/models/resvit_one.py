import torch
from collections import OrderedDict
from torch.autograd import Variable
import model.RFP_ResCNN.util.util as util
from model.RFP_ResCNN.util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import numpy as np
import cv2

class ResViT_model(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        
        self.schedulers = []
        self.optimizers = []
        
        # print original input and output channels
        print(f"Original input_nc: {opt.input_nc}, output_nc: {opt.output_nc}")
        
        # Generator와 Discriminator의 채널 수 설정
        self.input_nc = 3
        self.output_nc = 3
        
        # load/define networks
        self.netG = networks.define_G(
            self.input_nc,     # 3 channels input
            self.output_nc,    # 3 channels output
            opt.ngf,
            opt.which_model_netG,
            opt.vit_name,
            opt.fineSize,
            opt.pre_trained_path,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            self.gpu_ids,
            pre_trained_trans=opt.pre_trained_transformer,
            pre_trained_resnet=opt.pre_trained_resnet
        )

        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            
            # Discriminator는 input_nc + output_nc = 6 채널을 입력으로 받음
            self.netD = networks.define_D(
                6,  # 3(input_A) + 3(fake_B/real_B) channels
                opt.ndf,
                opt.which_model_netD,
                opt.vit_name,
                opt.fineSize,
                opt.n_layers_D,
                opt.norm,
                use_sigmoid,
                opt.init_type,
                self.gpu_ids
            )


        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        #print('---------- Networks initialized -------------')
        #networks.print_network(self.netG)
        #if self.isTrain:
        #    networks.print_network(self.netD)
        #print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        #input_B = input['B' if AtoB else 'A']
        
        # 입력 이미지를 3채널로 변환
        if input_A.size(1) == 1:
            input_A = input_A.repeat(1, 3, 1, 1)
        
        #if input_B.size(1) == 1:
        #    input_B = input_B.repeat(1, 3, 1, 1)
        
        # print(f"Input A shape: {input_A.size()}, Input B shape: {input_B.size()}")
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            #input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        
        self.input_A = input_A
        #self.input_B = input_B
        self.image_paths = input['A_paths']


    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)  # Generates single channel
        #self.real_B = Variable(self.input_B)  # Single channel (green)


    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            #self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # 디버그 출력
        # print(f"Real A shape: {self.real_A.size()}")
        # print(f"Fake B shape: {self.fake_B.size()}")
        # print(f"Real B shape: {self.real_B.size()}")
        
        # Fake pair - 3채널씩 연결
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        # print(f"Fake AB shape: {fake_AB.size()}")  # Should be [batch, 6, height, width]
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real pair - 3채널씩 연결
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # print(f"Real AB shape: {real_AB.size()}")  # Should be [batch, 6, height, width]
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_adv
        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())

                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im_batch(self.real_A.data)
        fake_B = util.tensor2im_batch(self.fake_B.data)
        #real_B = util.tensor2im(self.real_B.data)
        
        # print(f"Visualization shapes:")
        # print(f"real_A shape: {real_A.shape}")
        # print(f"fake_B shape: {fake_B.shape}")
        # print(f"real_B shape: {real_B.shape}")

        return OrderedDict([
            ('real_A', real_A),
            ('fake_B', fake_B),
            #('real_B', real_B)
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
