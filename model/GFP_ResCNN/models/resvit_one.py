import torch
from collections import OrderedDict
from torch.autograd import Variable
import model.GFP_ResCNN.util.util as util
from model.GFP_ResCNN.util.image_pool import ImagePool
from model.GFP_ResCNN.models.base_model import BaseModel
from model.GFP_ResCNN.models import networks
from torchvision import models
import numpy as np
import cv2
import torch.nn as nn 

class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # Initialize schedulers list at the beginning // 추가되었음
        self.schedulers = []
        self.optimizers = []
        
        # Modified: Set output_nc to 1 for green channel only // GFP 그린 채널만 쓰기 위해 수정된 코드
        opt.output_nc = 1

        self.spatial_attention = networks.SpatialAttention()
        
        # load/define networks
        self.netG = networks.define_G(
            opt.input_nc,  # input can be 3 channels # 현재 GFP는 1채널
            opt.output_nc, # output is 1 channel (Green)
            opt.ngf, # 첫 conv layer의 필터 개수
            opt.which_model_netG,
            opt.vit_name,
            opt.fineSize, # 1024
            opt.pre_trained_path,
            opt.norm, # batch
            not opt.no_dropout,
            opt.init_type, # normal
            self.gpu_ids,
            pre_trained_trans=opt.pre_trained_transformer,
            pre_trained_resnet=opt.pre_trained_resnet
        )

        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + 1,  # Modified: input_nc + 1 for concatenation
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

        # 학습 중단 후 재개 가능하도록 모델을 불러옴
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size) # 최근에 생성된 fake image 저장 버퍼 -> Generator가 생성한 Fake 이미지를 저장하고, Discriminator가 학습할 때 일부 과거 데이터를 섞어서 사용 // 현재 0이라 사용 안함
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = nn.L1Loss() # L1 loss 추가
            #self.criterionVGG = networks.VGGLoss() ## L1 -> VGG loss로 수정함 

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999) # betas : 1차 및 2차 모멘트 추정치를 제어 
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

    def set_input(self, input): #input -> data // dictionary로 돼있음 -> aligned dataset 확인해보기
        # AtoB = self.opt.which_direction == 'AtoB'
        
        input_A = input['A']
        # input_B = input['B' if AtoB else 'A']
        
        # # Extract green channel from input_B if it's RGB
        # if input_B.size(1) == 3:
        #     input_B = input_B[:, 1:2, :, :]  # Extract green channel only
            
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True) # non_blocking: 비동기로 전송 // CPU와 GPU가 동시에 작업을 수행 가능 → 속도 최적화
            #input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        
        self.input_A = input_A
        #self.input_B = input_B
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A) # Variable -> 자동 미분 계산 추적 // torch 최신 버전에서는 안써도 됨 requires_grad=True이면 자동으로 활성화됨
        #self.spatial_attention = self.spatial_attention.to(self.real_A.device)  # Attention 모듈 전체를 GPU로 이동
        #input_with_attention = self.spatial_attention(self.real_A) # 이거 내가 추가함 
        
        self.fake_B = self.netG(self.real_A)  # Generates single channel #원래 self.read_A들어가야함
        self.real_B = Variable(self.input_B)  # Single channel (green)함


    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A) 
            #self.spatial_attention = self.spatial_attention.to(self.real_A.device)  # Attention 모듈 전체를 GPU로 이동
            #input_with_attention = self.spatial_attention(self.real_A) # 이거 내가 추가함 
            
            self.fake_B = self.netG(self.real_A)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach()) # detach : Generator의 역전파가 영향을 받지 않도록 차단 // discriminator만 학습
        self.loss_D_fake = self.criterionGAN(pred_fake, False) # fake image를 가짜라고 판별해야 정답 // 가짜인데 진짜라고 판별하면 손실값이 증가

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)#*3

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv # 두 손실 값 평균

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv#*5 # fake image가 진짜라고 판별하도록 유도하는 함수 // 만약 가짜라고 예측 시 손실값 증가
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)*self.opt.lambda_A#*15
        #self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B)#*50
        #self.loss_G_SSIM = networks.ssim_loss(self.fake_B, self.real_B)#*30
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*1# + self.loss_G_SSIM + self.loss_G_VGG  
        
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
                            #('G_VGG', self.loss_G_VGG.item()), 
                            #('G_SSIM', self.loss_G_SSIM.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            
                            ])

    def get_current_visuals(self):
        real_A_batch = util.tensor2im_batch(self.real_A.data)  # [(H, W, 3), ...]
        fake_B_batch = util.tensor2im_batch(self.fake_B.data)  # [(H, W) or (H, W, 1), ...]
    
        fake_B_green_batch = []
    
        for fake_B_temp in fake_B_batch:
            H, W = fake_B_temp.shape[:2]
            fake_B_green = np.zeros((H, W, 3), dtype=np.uint8)
            
            if len(fake_B_temp.shape) == 2:
                fake_B_green[:, :, 1] = fake_B_temp  # gray -> green
            elif fake_B_temp.shape[2] == 1:
                fake_B_green[:, :, 1] = fake_B_temp[:, :, 0]
            else:
                fake_B_green[:, :, 1] = fake_B_temp[:, :, 0]  # RGB라면 그냥 첫 채널
    
            fake_B_green_batch.append(fake_B_green)
    
        return OrderedDict([
            ('real_A', real_A_batch),
            ('fake_B', fake_B_green_batch)
        ])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

