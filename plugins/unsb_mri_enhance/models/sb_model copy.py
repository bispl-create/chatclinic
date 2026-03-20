import os
import numpy as np
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

class SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for SB model
        """
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
        parser.add_argument('--lambda_mask_trimap', type=float, default=0.0, help='weight for trimap-based input/output mask consistency')
        parser.add_argument('--lambda_mask_nsd', type=float, default=0.0, help='weight for one-sided NSD-like boundary consistency')
        parser.add_argument('--mask_debug_epochs', type=int, default=0, help='number of epochs to save debug masks once per epoch (0 disables)')
        parser.add_argument('--mask_debug_epoch_interval', type=int, default=1, help='save debug masks every N epochs in epoch mode')
        parser.add_argument('--mask_debug_steps', type=int, default=0, help='number of first training steps to save per epoch (0 disables)')
        parser.add_argument('--mask_debug_interval', type=int, default=1, help='save debug masks every N steps')
        parser.add_argument('--mask_debug_dir', type=str, default='', help='directory to save debug masks (defaults to <checkpoints_dir>/<name>/debug_masks)')
        parser.add_argument('--trimap_radius', type=int, default=3, help='trimap radius (r)')
        parser.add_argument('--mask_extract_threshold', type=float, default=1e-3, help='threshold on denormalized [0,1] intensity for non-zero mask extraction')
        parser.add_argument('--mask_extract_scale', type=float, default=1e-2, help='deprecated (kept for compatibility)')
        parser.add_argument('--mask_nsd_t', type=float, default=3.0, help='tolerance distance (pixels) for one-sided NSD-like loss')
        parser.add_argument('--mask_nsd_gamma', type=float, default=2.0, help='softness of distance gate for one-sided NSD-like loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        # DMD2 arguments
        parser.add_argument('--use_dmd', type=util.str2bool, nargs='?', const=True, default=False, help='Use DMD2 distributional matching')
        parser.add_argument('--adm_ckpt_path', type=str, default=None, help='Path to pretrained ADM checkpoint')
        parser.add_argument('--adm_fake_ckpt_path', type=str, default=None,
                            help='Optional path to a separate ADM checkpoint for fake_unet initialization')
        parser.add_argument('--adm_num_channels', type=int, default=128, help='ADM UNet base channel width')
        parser.add_argument('--adm_num_head_channels', type=int, default=-1, help='ADM UNet attention head channels')
        parser.add_argument('--adm_attention_resolutions', type=str, default='16,8', help='ADM UNet attention resolutions')
        parser.add_argument('--adm_resblock_updown', type=util.str2bool, nargs='?', const=True, default=False, help='Use residual up/down blocks in ADM UNet')
        parser.add_argument('--adm_use_scale_shift_norm', type=util.str2bool, nargs='?', const=True, default=True, help='Use scale-shift normalization in ADM UNet')
        parser.add_argument('--adm_learn_sigma', type=util.str2bool, nargs='?', const=True, default=False, help='Enable ADM learn_sigma output')
        parser.add_argument('--lambda_DM', type=float, default=1.0, help='Weight for distribution matching loss')
        parser.add_argument('--lambda_CLS', type=float, default=0.1, help='Weight for classifier adversarial loss')
        parser.add_argument('--guidance_update_ratio', type=int, default=1, help='Deprecated: guidance is always updated every step')
        parser.add_argument('--dfake_gen_update_ratio', type=int, default=1, help='Update generator once every N steps while guidance can be updated more frequently')
        parser.add_argument('--num_train_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
        parser.add_argument('--min_step_percent', type=float, default=0.02, help='Min timestep percent for DM loss')
        parser.add_argument('--max_step_percent', type=float, default=0.98, help='Max timestep percent for DM loss')
        parser.add_argument('--dm_denom_min', type=float, default=1e-3, help='Clamp minimum for DM normalization denominator')
        parser.add_argument('--dm_grad_clip', type=float, default=10.0, help='Clip absolute DM gradient value; <=0 disables clipping')
        parser.add_argument('--dmd_warmup_steps', type=int, default=0,
                            help='Number of DMD update steps to linearly warm up DMD losses (0 disables warmup)')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.mode.lower() == "sb":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. 
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB']
        if opt.lambda_mask_trimap > 0.0:
            self.loss_names += ['Mask_trimap']
        if opt.lambda_mask_nsd > 0.0:
            self.loss_names += ['Mask_NSD']
        if opt.use_dmd:
            self.loss_names += ['DM', 'G_CLS', 'Guidance_CLS', 'Fake']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        if opt.lambda_mask_trimap > 0.0 or opt.lambda_mask_nsd > 0.0:
            self.visual_names += ['mask_in', 'mask_fake']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D','E']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

            # DMD2 guidance model
            if opt.use_dmd:
                from .adm_wrapper import ADMWrapper
                from .dmd_guidance import DMDGuidance

                sample_size = getattr(opt, 'crop_size', 256)
                self.adm_model = ADMWrapper(
                    model_path=opt.adm_ckpt_path,
                    image_size=sample_size,
                    num_channels=opt.adm_num_channels,
                    attention_resolutions=opt.adm_attention_resolutions,
                    num_head_channels=opt.adm_num_head_channels,
                    use_scale_shift_norm=opt.adm_use_scale_shift_norm,
                    resblock_updown=opt.adm_resblock_updown,
                    learn_sigma=opt.adm_learn_sigma,
                    num_train_timesteps=opt.num_train_timesteps,
                )

                # Move to device
                self.adm_model = self.adm_model.to(self.device)

                fake_adm_wrapper = None
                if opt.adm_fake_ckpt_path is not None:
                    if not os.path.exists(opt.adm_fake_ckpt_path):
                        raise FileNotFoundError(f"adm_fake_ckpt_path not found: {opt.adm_fake_ckpt_path}")
                    print(f"Loading separate fake ADM checkpoint from {opt.adm_fake_ckpt_path}")
                    fake_adm_wrapper = ADMWrapper(
                        model_path=opt.adm_fake_ckpt_path,
                        image_size=sample_size,
                        num_channels=opt.adm_num_channels,
                        attention_resolutions=opt.adm_attention_resolutions,
                        num_head_channels=opt.adm_num_head_channels,
                        use_scale_shift_norm=opt.adm_use_scale_shift_norm,
                        resblock_updown=opt.adm_resblock_updown,
                        learn_sigma=opt.adm_learn_sigma,
                        num_train_timesteps=opt.num_train_timesteps,
                    ).to(self.device)

                # Create DMD guidance (mu_fake + classifier)
                self.netGuidance = DMDGuidance(
                    opt,
                    self.adm_model,
                    fake_adm_wrapper=fake_adm_wrapper
                ).to(self.device)
                self.netGuidance_fake = self.netGuidance.fake_unet
                self.netGuidance_cls = self.netGuidance.cls_pred_branch

                # Optimizer for guidance model (fake_unet + classifier if enabled)
                guidance_params = list(self.netGuidance.fake_unet.parameters())
                if self.netGuidance.cls_pred_branch is not None:
                    guidance_params += list(self.netGuidance.cls_pred_branch.parameters())

                self.optimizer_Guidance = torch.optim.Adam(
                    guidance_params,
                    lr=5e-6, betas=(opt.beta1, opt.beta2)
                )
                self.optimizers.append(self.optimizer_Guidance)

                # Tracking iteration for guidance update ratio
                self.iteration = 0

                # Register DMD sub-networks for checkpoint save/load with BaseModel.
                self.model_names += ['Guidance_fake']
                if self.netGuidance_cls is not None:
                    self.model_names += ['Guidance_cls']
        self.dmd_warmup_steps = int(getattr(opt, 'dmd_warmup_steps', 0))
        self.mask_in = None
        self.mask_fake = None
        self._mask_debug_step = 0
        self._mask_debug_epoch = -1
        self._mask_debug_epoch_index = 0
        self._mask_debug_saved_this_epoch = False
        self._mask_debug_dir = opt.mask_debug_dir.strip()
        if self._mask_debug_dir == '':
            self._mask_debug_dir = os.path.join(self.save_dir, 'debug_masks')
        os.makedirs(self._mask_debug_dir, exist_ok=True)

    def on_epoch_start(self, epoch):
        epoch = int(epoch)
        if self._mask_debug_epoch != epoch:
            self._mask_debug_epoch = epoch
            self._mask_debug_epoch_index += 1
            self._mask_debug_saved_this_epoch = False
            self._mask_debug_step = 0

    def _get_dmd_warmup_factor(self):
        warmup_steps = max(int(getattr(self.opt, 'dmd_warmup_steps', 0)), 0)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, self.iteration / float(warmup_steps))
            
    def data_dependent_initialize(self, data,data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data,data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()  
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self._maybe_save_mask_debug()
        compute_generator_update = (not self.opt.use_dmd) or (self.iteration % self.opt.dfake_gen_update_ratio == 0)

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        # update Guidance (DMD2: fake_unet + classifier)
        if self.opt.use_dmd:
            # Keep teacher frozen; only train fake_unet (+ optional cls branch).
            self.netGuidance.real_unet.freeze_parameters()
            self.set_requires_grad(self.netGuidance.fake_unet, True)
            if self.netGuidance.cls_pred_branch is not None:
                self.set_requires_grad(self.netGuidance.cls_pred_branch, True)
            self.optimizer_Guidance.zero_grad()
            self.loss_Guidance = self.compute_Guidance_loss()
            self.loss_Guidance.backward()
            self.optimizer_Guidance.step()

        # update G (optionally less frequent than guidance, DMD2-style)
        if compute_generator_update:
            self.set_requires_grad(self.netD, False)
            self.set_requires_grad(self.netE, False)
            if self.opt.use_dmd:
                # Keep teacher frozen. fake_unet/cls must keep requires_grad=True
                # because guided-diffusion attention checkpointing expects params
                # passed to autograd.grad to require grad.
                self.netGuidance.real_unet.freeze_parameters()
                self.set_requires_grad(self.netGuidance.fake_unet, True)
                if self.netGuidance.cls_pred_branch is not None:
                    self.set_requires_grad(self.netGuidance.cls_pred_branch, True)

            self.optimizer_G.zero_grad()
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad()
            self.loss_G = self.compute_G_loss()
            self.loss_G.backward()
            self.optimizer_G.step()
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.step()
        else:
            # Keep last generator-side losses for logging on non-generator steps.
            self.loss_G = getattr(self, 'loss_G', torch.tensor(0.0, device=self.device))
            self.loss_G_GAN = getattr(self, 'loss_G_GAN', torch.tensor(0.0, device=self.device))
            self.loss_SB = getattr(self, 'loss_SB', torch.tensor(0.0, device=self.device))
            self.loss_NCE = getattr(self, 'loss_NCE', torch.tensor(0.0, device=self.device))
            if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
                self.loss_NCE_Y = getattr(self, 'loss_NCE_Y', torch.tensor(0.0, device=self.device))
            self.loss_Mask_trimap = getattr(self, 'loss_Mask_trimap', torch.tensor(0.0, device=self.device))
            self.loss_Mask_NSD = getattr(self, 'loss_Mask_NSD', torch.tensor(0.0, device=self.device))
            if self.opt.use_dmd:
                self.loss_DM = getattr(self, 'loss_DM', torch.tensor(0.0, device=self.device))
                self.loss_G_CLS = getattr(self, 'loss_G_CLS', torch.tensor(0.0, device=self.device))

        # Increment iteration counter for DMD
        if self.opt.use_dmd:
            self.iteration += 1       
        
    def set_input(self, input,input2=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.mask_in = None
        self.mask_fake = None

    def forward(self):
        
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times])
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]
        
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):
                
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_1     = self.netG(Xt, time_idx, z)
                
                Xt2       = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_12    = self.netG(Xt2, time_idx, z)
                
                
                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1B = self.netG(XtB, time_idx, z)
            if self.opt.nce_idt:
                self.XtB = XtB.detach()
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()
                      
        
        z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
        z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])
        
        self.fake = self.netG(self.realt,self.time_idx,z_in)
        self.fake_B2 =  self.netG(self.real_A_noisy2,self.time_idx,z_in2)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        if self.opt.lambda_mask_trimap > 0.0 or self.opt.lambda_mask_nsd > 0.0 or self.opt.mask_debug_steps > 0 or self.opt.mask_debug_epochs > 0:
            self.mask_in = self._get_input_mask(self.real_A)
            self.mask_fake = self._image_to_soft_mask(self.fake_B)
            
        if self.opt.phase == 'test':
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1     = self.netG(Xt, time_idx, z)
                    
                    setattr(self, "fake_"+str(t+1), Xt_1)
                    
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs =  self.real_A.size(0)
        
        fake = self.fake_B.detach()
        std = torch.rand(size=[1]).item() * self.opt.std
        
        pred_fake = self.netD(fake,self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B,self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def compute_E_loss(self):
        
        bs =  self.real_A.size(0)
        
        """Calculate GAN loss for the discriminator"""
        
        XtXt_1 = torch.cat([self.real_A_noisy,self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2,self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        
        return self.loss_E

    def compute_Guidance_loss(self):
        """
        Calculate loss for DMD Guidance model (fake_unet + classifier)
        Trains mu_fake to denoise and classifier to distinguish real/fake
        IMPORTANT: fake_B must be detached to prevent gradients flowing to Generator
        """
        loss_dict, log_dict = self.netGuidance(
            clean_images=self.fake_B.detach(),
            real_images=self.real_B,
            generator_turn=False,
            guidance_turn=True,
        )

        self.loss_Fake = loss_dict['loss_fake_mean']
        if self.opt.lambda_CLS > 0 and 'guidance_cls_loss' in loss_dict:
            self.loss_Guidance_CLS = loss_dict['guidance_cls_loss']
        else:
            self.loss_Guidance_CLS = torch.tensor(0.0, device=self.device)

        # Total guidance loss (no warmup)
        total_loss = self.loss_Fake + self.opt.lambda_CLS * self.loss_Guidance_CLS

        # Debug stats for fake denoising behavior.
        self.fake_pred_noise_mean = log_dict.get('faketrain_pred_noise_mean', torch.tensor(0.0, device=self.device))
        self.fake_pred_noise_std = log_dict.get('faketrain_pred_noise_std', torch.tensor(0.0, device=self.device))
        self.fake_target_noise_mean = log_dict.get('faketrain_target_noise_mean', torch.tensor(0.0, device=self.device))
        self.fake_target_noise_std = log_dict.get('faketrain_target_noise_std', torch.tensor(0.0, device=self.device))
        self.fake_noise_mae = log_dict.get('faketrain_noise_mae', torch.tensor(0.0, device=self.device))

        return total_loss

    def compute_G_loss(self):
        bs =  self.real_A.size(0)
        tau = self.opt.tau
        
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std
        
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake,self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() 
        else:
            self.loss_G_GAN = 0.0
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            
            bs = self.opt.batch_size

            ET_XY    = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY
            self.loss_SB += self.opt.tau*torch.mean((self.real_A_noisy-self.fake_B)**2)
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_Mask_trimap = torch.tensor(0.0, device=self.device)
        self.loss_Mask_NSD = torch.tensor(0.0, device=self.device)
        if self.opt.lambda_mask_trimap > 0.0 or self.opt.lambda_mask_nsd > 0.0:
            self.loss_Mask_trimap, self.loss_Mask_NSD = self.compute_mask_loss()
        loss_mask = self.opt.lambda_mask_trimap * self.loss_Mask_trimap + self.opt.lambda_mask_nsd * self.loss_Mask_NSD

        # DMD2 Distribution Matching Loss
        if self.opt.use_dmd:
            loss_dict, _ = self.netGuidance(
                clean_images=self.fake_B,
                real_images=self.real_B,
                generator_turn=True,
                guidance_turn=False,
            )

            self.loss_DM = loss_dict['loss_dm']
            
            if self.opt.lambda_CLS > 0 and 'gen_cls_loss' in loss_dict:
                self.loss_G_CLS = loss_dict['gen_cls_loss']
            else:
                self.loss_G_CLS = torch.tensor(0.0, device=self.device)

            dmd_warmup_factor = self._get_dmd_warmup_factor()
            self.dmd_warmup_factor = dmd_warmup_factor
            # Add DM loss to total generator loss
            self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both + \
                          self.opt.lambda_DM * self.loss_DM * dmd_warmup_factor + \
                          self.loss_Mask_trimap * self.opt.lambda_mask_trimap + self.loss_Mask_NSD * self.opt.lambda_mask_nsd + \
                          self.opt.lambda_CLS * self.loss_G_CLS * dmd_warmup_factor
        else:
            self.loss_G = self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both + loss_mask

        return self.loss_G


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        feat_k = self.netG(src, self.time_idx*0,z,self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def _to_single_channel(self, x):
        x = x.to(self.device)
        if x.dim() == 4 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        return x

    def _image_to_soft_mask(self, img):
        # Dataset tensors are normalized to [-1, 1], so background(0 in image space)
        # becomes -1. Build mask from denormalized [0,1] values.
        x = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        x01 = ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
        if x01.dim() == 4 and x01.size(1) > 1:
            x01 = x01.amax(dim=1, keepdim=True)
        threshold = max(float(self.opt.mask_extract_threshold), 0.0)
        return (x01 > threshold).float()

    def _get_input_mask(self, ref_img):
        return self._image_to_soft_mask(ref_img)

    def _trimap_masks(self, mask_in):
        r = int(max(self.opt.trimap_radius, 0))
        if r <= 0:
            return torch.zeros_like(mask_in), torch.zeros_like(mask_in)
        k = 2 * r + 1
        core = -F.max_pool2d(-mask_in, kernel_size=k, stride=1, padding=r)
        bg = 1.0 - F.max_pool2d(mask_in, kernel_size=k, stride=1, padding=r)
        return core.clamp(0.0, 1.0), bg.clamp(0.0, 1.0)

    def _mask_boundary(self, mask):
        # approx image boundary from a soft mask
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.device, dtype=mask.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=self.device, dtype=mask.dtype).view(1, 1, 3, 3)
        gx = F.conv2d(mask, kx, padding=1)
        gy = F.conv2d(mask, ky, padding=1)
        b = torch.sqrt(gx * gx + gy * gy)
        b = b / (b.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return b

    def _distance_to_in_boundary(self, mask_in):
        # O(HW) iterative approx of distance transform (for small t)
        bin_mask = (mask_in > 0.5).float()
        dil = F.max_pool2d(bin_mask, 3, stride=1, padding=1)
        ero = -F.max_pool2d(-bin_mask, 3, stride=1, padding=1)
        boundary = (dil - ero).clamp(min=0.0, max=1.0)
        max_d = int(max(float(self.opt.mask_nsd_t), 1.0))
        dist = torch.full_like(mask_in, float(max_d))
        frontier = boundary > 0.0
        visited = frontier.clone()
        dist = torch.where(frontier, torch.zeros_like(dist), dist)
        for d in range(1, max_d + 1):
            frontier = F.max_pool2d(frontier.float(), 3, stride=1, padding=1) > 0.5
            new = frontier & (~visited)
            dist = torch.where(new, torch.full_like(dist, float(d)), dist)
            visited = visited | frontier
        return dist

    def _save_mask_png(self, tensor_mask, path):
        # tensor_mask: [B,1,H,W] in [0,1]
        m = tensor_mask[0].detach().clamp(0.0, 1.0).float().cpu()
        if m.dim() == 3:
            m = m[0:1]
        m_rgb = (m * 255.0).byte().permute(1, 2, 0).repeat(1, 1, 3).numpy()
        util.save_image(m_rgb, path)

    def _maybe_save_mask_debug(self):
        if not self.isTrain:
            return
        if self.mask_in is None or self.mask_fake is None:
            return

        max_epochs = int(getattr(self.opt, 'mask_debug_epochs', 0))
        if max_epochs > 0:
            if self._mask_debug_epoch_index <= 0:
                self._mask_debug_epoch_index = 1
            if self._mask_debug_epoch_index > max_epochs:
                return
            epoch_interval = max(int(getattr(self.opt, 'mask_debug_epoch_interval', 1), 1))
            if (self._mask_debug_epoch_index - 1) % epoch_interval != 0:
                return
            if self._mask_debug_saved_this_epoch:
                return
            epoch_tag = self._mask_debug_epoch if self._mask_debug_epoch >= 0 else self._mask_debug_epoch_index
            self._save_mask_png(self.mask_in, os.path.join(self._mask_debug_dir, f'epoch{epoch_tag:04d}_in.png'))
            self._save_mask_png(self.mask_fake, os.path.join(self._mask_debug_dir, f'epoch{epoch_tag:04d}_out.png'))
            self._mask_debug_saved_this_epoch = True
            return

        max_steps = int(self.opt.mask_debug_steps)
        if max_steps <= 0:
            return
        if self._mask_debug_step >= max_steps:
            return
        interval = max(int(self.opt.mask_debug_interval), 1)
        self._mask_debug_step += 1
        if (self._mask_debug_step - 1) % interval != 0:
            return

        step = self._mask_debug_step
        epoch_tag = self._mask_debug_epoch if self._mask_debug_epoch >= 0 else 0
        self._save_mask_png(self.mask_in, os.path.join(self._mask_debug_dir, f'epoch{epoch_tag:04d}_step{step:06d}_in.png'))
        self._save_mask_png(self.mask_fake, os.path.join(self._mask_debug_dir, f'epoch{epoch_tag:04d}_step{step:06d}_out.png'))

    def compute_mask_loss(self):
        if self.mask_in is None or self.mask_fake is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        eps = 1e-6
        mask_in = torch.nan_to_num(self.mask_in.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        mask_fake = torch.nan_to_num(self.mask_fake.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        mask_fake_prob = mask_fake.clamp(eps, 1.0 - eps)

        loss_trimap = torch.tensor(0.0, device=self.device)
        if self.opt.lambda_mask_trimap > 0.0:
            core, bg = self._trimap_masks(mask_in)
            region = core + bg
            if region.sum() < eps:
                loss_trimap = torch.tensor(0.0, device=self.device)
            else:
                loss_core = F.binary_cross_entropy(mask_fake_prob, torch.ones_like(mask_fake_prob), weight=core, reduction='sum')
                loss_bg = F.binary_cross_entropy(mask_fake_prob, torch.zeros_like(mask_fake_prob), weight=bg, reduction='sum')
                loss_trimap = (loss_core + loss_bg) / (region.sum() + eps)

        loss_nsd = torch.tensor(0.0, device=self.device)
        if self.opt.lambda_mask_nsd > 0.0:
            dist = self._distance_to_in_boundary(mask_in)
            gate = torch.sigmoid((self.opt.mask_nsd_t - dist) / max(self.opt.mask_nsd_gamma, 1e-6))
            b_out = self._mask_boundary(mask_fake)
            precision = (b_out * gate).sum(dim=(2, 3)) / (b_out.sum(dim=(2, 3)) + eps)
            loss_nsd = (1.0 - precision).mean()

        return loss_trimap, loss_nsd

    def load_networks(self, epoch):
        """
        Override load to tolerate missing DMD-side checkpoints when resuming from
        older experiments that do not contain guidance subnetworks.
        """
        for name in self.model_names:
            if not isinstance(name, str):
                continue

            load_filename = '%s_net_%s.pth' % (epoch, name)
            if self.opt.isTrain and self.opt.pretrained_name is not None:
                load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
            else:
                load_dir = self.save_dir

            load_path = os.path.join(load_dir, load_filename)
            if not os.path.exists(load_path):
                if name.startswith('Guidance_'):
                    print('[warn] missing %s; skipped for guidance resume. Using current initialization.' % load_filename)
                    continue
                if name.startswith('DMD_'):
                    print('[warn] missing %s; skipped for DMD resume. Using ADM init for DMD model initialization.' % load_filename)
                    continue
                raise FileNotFoundError('[error] missing checkpoint: %s' % load_path)

            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = self._safe_torch_load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            target_state = net.state_dict()
            filtered_state = {}
            skipped_by_shape = []

            for key, value in state_dict.items():
                if key not in target_state:
                    continue
                target_value = target_state[key]
                if torch.is_tensor(value) and torch.is_tensor(target_value) and value.shape == target_value.shape:
                    filtered_state[key] = value
                else:
                    skipped_by_shape.append((key, tuple(value.shape), tuple(target_value.shape) if torch.is_tensor(target_value) else 'non-tensor'))

            missing, unexpected = net.load_state_dict(filtered_state, strict=False)
            if skipped_by_shape:
                print('[warn] skipped %d keys due to shape mismatch while loading %s' % (len(skipped_by_shape), load_filename))
                for key, ckpt_shape, model_shape in skipped_by_shape[:20]:
                    print('  - %s: ckpt%s != model%s' % (key, ckpt_shape, model_shape))
            if missing:
                print('[warn] missing keys while loading %s: %s' % (load_filename, missing))
            if unexpected:
                print('[warn] unexpected keys while loading %s: %s' % (load_filename, unexpected))
