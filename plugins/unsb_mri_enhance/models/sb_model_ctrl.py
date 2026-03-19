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
        """  Configures options specific for SB model (DMD2 + ControlNet)
        """
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')

        parser.add_argument('--lambda_GAN', type=float, default=0.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
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

        # ControlNet arguments
        parser.add_argument('--control_mode', type=str, default='none', choices=['none', 'controlnet'],
                            help='none: original UNSB training. controlnet: freeze G and train ControlNet.')
        parser.add_argument('--pretrained_G_path', type=str, default=None,
                            help='Path to pretrained generator weights for ControlNet post-training.')
        parser.add_argument('--pretrained_F_path', type=str, default=None)
        parser.add_argument('--pretrained_D_path', type=str, default=None)
        parser.add_argument('--pretrained_E_path', type=str, default=None)
        parser.add_argument('--pretrained_Guidance_path', type=str, default=None,
                            help='Path to pretrained DMD2 fake_unet weights to warm-start guidance training.')
        parser.add_argument('--pretrained_Guidance_cls_path', type=str, default=None,
                            help='Path to pretrained DMD2 cls_pred_branch weights to warm-start guidance training.')
        parser.add_argument('--auto_load_DE', type=util.str2bool, nargs='?', const=True, default=True,
                            help='Auto-load D/E from same folder as pretrained_G_path.')
        parser.add_argument('--freeze_G', type=util.str2bool, nargs='?', const=True, default=True,
                            help='Freeze generator weights (recommended for ControlNet).')
        parser.add_argument('--freeze_F', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--train_F', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--lr_C', type=float, default=None, help='Learning rate for ControlNet (default: same as --lr).')
        # Injection ports + gating schedule
        parser.add_argument('--ctrl_ports', type=str, default='low', choices=['low', 'mid', 'high'])
        parser.add_argument('--ctrl_scale', type=float, default=1.0)
        parser.add_argument('--gate_down', type=str, default='linear', choices=['off', 'const', 'linear', 'pow', 'step'])
        parser.add_argument('--gate_mid', type=str, default='pow', choices=['off', 'const', 'linear', 'pow', 'step'])
        parser.add_argument('--gate_up', type=str, default='pow', choices=['off', 'const', 'linear', 'pow', 'step'])
        parser.add_argument('--gate_down_pow', type=float, default=1.0)
        parser.add_argument('--gate_mid_pow', type=float, default=2.0)
        parser.add_argument('--gate_up_pow', type=float, default=3.0)
        parser.add_argument('--gate_step_start', type=int, default=2)
        # Condition channels
        parser.add_argument('--cond_use_lowpass', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--cond_use_grad', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--cond_use_boundary', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--cond_ref', type=str, default='x0', choices=['x0', 'xt'])
        parser.add_argument('--mask_key', type=str, default='')
        parser.add_argument('--mask_threshold', type=float, default=0.01)
        parser.add_argument('--boundary_width', type=int, default=3)
        parser.add_argument('--low_sigma', type=float, default=1.5)
        # Extra losses
        parser.add_argument('--lambda_freq', type=float, default=0.0)
        parser.add_argument('--freq_w_low', type=float, default=0.0)
        parser.add_argument('--freq_w_high', type=float, default=1.0)
        parser.add_argument('--freq_cutoff', type=float, default=0.15)
        parser.add_argument('--lambda_gram', type=float, default=0.0)
        parser.add_argument('--gram_layers', type=str, default='1,2,3')
        parser.add_argument('--gram_start_idx', type=int, default=3)
        parser.add_argument('--lambda_bd', type=float, default=0.0)
        parser.add_argument('--loss_ref', type=str, default='x0', choices=['x0', 'xt'])
        parser.add_argument('--dmd_update_ctrl', type=util.str2bool, nargs='?', const=True, default=True,
                            help='If False, DMD DM/CLS loss gradients are blocked from flowing to ControlNet (fake_B detached).')

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

        # ControlNet-friendly defaults
        if getattr(opt, 'control_mode', 'none') == 'controlnet':
            parser.set_defaults(
                freeze_G=True, freeze_F=True, train_F=False,
                ctrl_ports='low', ctrl_scale=1.0,
                gate_down='linear', gate_mid='pow', gate_up='pow',
                gate_down_pow=1.0, gate_mid_pow=2.0, gate_up_pow=3.0,
                gate_step_start=2,
                cond_use_lowpass=True, cond_use_grad=True, cond_use_boundary=True,
                cond_ref='x0', loss_ref='x0',
                lambda_freq=1.0, lambda_gram=0.05, lambda_bd=2.0,
                freq_w_low=0.0, freq_w_high=1.0, freq_cutoff=0.15,
                low_sigma=1.5, gram_layers='1,2,3', gram_start_idx=3, boundary_width=3,
            )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.use_control = (getattr(opt, 'control_mode', 'none') == 'controlnet')
        self.train_F = (not self.use_control) or bool(getattr(opt, 'train_F', False))

        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SB']
        if opt.use_dmd:
            self.loss_names += ['DM', 'G_CLS', 'Guidance_CLS', 'Fake']
        if getattr(opt, 'lambda_freq', 0.0) > 0:
            self.loss_names += ['freq', 'freq_low', 'freq_high']
        if getattr(opt, 'lambda_gram', 0.0) > 0:
            self.loss_names += ['gram']
        if getattr(opt, 'lambda_bd', 0.0) > 0:
            self.loss_names += ['bd']

        self.visual_names = ['real_A', 'real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'E']
            if self.use_control:
                self.model_names = ['G', 'C', 'F', 'D', 'E']
        else:
            self.model_names = ['G']
            if self.use_control:
                self.model_names = ['G', 'C']

        # define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        # ControlNet
        self.cond_nc = 0
        if self.use_control:
            self.cond_nc = (
                (opt.input_nc if opt.cond_use_lowpass else 0) +
                (1 if opt.cond_use_grad else 0) +
                (1 if opt.cond_use_boundary else 0)
            )
            if self.cond_nc <= 0:
                raise ValueError('control_mode=controlnet requires at least one cond_* flag to be True.')
            self.netC = networks.define_C(
                cond_nc=self.cond_nc, ngf=opt.ngf, norm=opt.normG,
                init_type=opt.init_type, init_gain=opt.init_gain,
                gpu_ids=self.gpu_ids, opt=opt
            )

        # Load pretrained weights (ControlNet post-training)
        continue_train = bool(getattr(opt, 'continue_train', False))
        pretrained_name = getattr(opt, 'pretrained_name', None)
        if self.use_control and getattr(opt, 'pretrained_G_path', None) is not None and not continue_train and pretrained_name is None:
            self._load_pretrained(self.netG, opt.pretrained_G_path, name='G')
        if self.use_control and getattr(opt, 'pretrained_F_path', None) is not None and not continue_train and pretrained_name is None:
            self._load_pretrained(self.netF, opt.pretrained_F_path, name='F')

        # Freeze G / F
        if self.use_control and getattr(opt, 'freeze_G', True):
            self.set_requires_grad(self.netG, False)
            self.netG.eval()
        if self.use_control and getattr(opt, 'freeze_F', True) and not self.train_F:
            self.set_requires_grad(self.netF, False)
            self.netF.eval()

        # Kernel caches for condition building
        self._gauss_cache = {}
        self._rfft_mask_cache = {}
        self._sobel_cache = {}

        # Monitoring scalars
        self.loss_freq_low = 0.0
        self.loss_freq_high = 0.0
        self.ctrl_norm_down = 0.0
        self.ctrl_norm_mid = 0.0
        self.ctrl_norm_up = 0.0

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # Pretrained D/E warm-start (ControlNet mode)
            if self.use_control and not continue_train and pretrained_name is None:
                def _guess_from_G(token):
                    if getattr(opt, 'pretrained_G_path', None) is None:
                        return None
                    base = os.path.basename(opt.pretrained_G_path)
                    if 'net_G' in base:
                        return os.path.join(os.path.dirname(opt.pretrained_G_path),
                                            base.replace('net_G', f'net_{token}'))
                    return None

                d_path = getattr(opt, 'pretrained_D_path', None)
                e_path = getattr(opt, 'pretrained_E_path', None)
                if getattr(opt, 'auto_load_DE', True):
                    if d_path is None:
                        d_path = _guess_from_G('D')
                    if e_path is None:
                        e_path = _guess_from_G('E')
                if d_path and os.path.isfile(d_path):
                    self._load_pretrained(self.netD, d_path, name='D')
                elif d_path:
                    print(f'[SBModel] Warning: pretrained D not found: {d_path}')
                if e_path and os.path.isfile(e_path):
                    self._load_pretrained(self.netE, e_path, name='E')
                elif e_path:
                    print(f'[SBModel] Warning: pretrained E not found: {e_path}')

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            # D, E always train
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

            # G optimizer — skip if frozen in ControlNet mode
            self.optimizer_G = None
            if (not self.use_control) or (not getattr(opt, 'freeze_G', True)):
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_G)

            # ControlNet optimizer
            self.optimizer_C = None
            if self.use_control:
                lr_C = opt.lr if getattr(opt, 'lr_C', None) is None else opt.lr_C
                self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=lr_C, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_C)

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
                self.adm_model = self.adm_model.to(self.device)
                self.netGuidance = DMDGuidance(opt, self.adm_model).to(self.device)

                # Register DMD sub-nets so BaseModel.save/load_networks handles them
                self.netDMD_fake_unet = self.netGuidance.fake_unet
                self.model_names.append('DMD_fake_unet')
                if self.netGuidance.cls_pred_branch is not None:
                    self.netDMD_cls_head = self.netGuidance.cls_pred_branch
                    self.model_names.append('DMD_cls_head')

                # Warm-start fake_unet from a pre-trained DMD2 checkpoint if provided
                guidance_ckpt = getattr(opt, 'pretrained_Guidance_path', None)
                if guidance_ckpt and os.path.isfile(guidance_ckpt) and not continue_train and pretrained_name is None:
                    self._load_pretrained(self.netGuidance.fake_unet, guidance_ckpt, name='Guidance')
                elif guidance_ckpt:
                    print(f'[SBModel] Warning: pretrained_Guidance_path not found: {guidance_ckpt}')

                cls_ckpt = getattr(opt, 'pretrained_Guidance_cls_path', None)
                if cls_ckpt and os.path.isfile(cls_ckpt) and not continue_train and pretrained_name is None:
                    if self.netGuidance.cls_pred_branch is not None:
                        self._load_pretrained(self.netGuidance.cls_pred_branch, cls_ckpt, name='Guidance_cls')
                    else:
                        print(f'[SBModel] Warning: pretrained_Guidance_cls_path given but cls_pred_branch is None')
                elif cls_ckpt:
                    print(f'[SBModel] Warning: pretrained_Guidance_cls_path not found: {cls_ckpt}')

                guidance_params = list(self.netGuidance.fake_unet.parameters())
                if self.netGuidance.cls_pred_branch is not None:
                    guidance_params += list(self.netGuidance.cls_pred_branch.parameters())
                self.optimizer_Guidance = torch.optim.Adam(
                    guidance_params, lr=5e-6, betas=(opt.beta1, opt.beta2)
                )
                self.optimizers.append(self.optimizer_Guidance)
                self.iteration = 0

    def data_dependent_initialize(self, data, data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()
            if self.opt.lambda_NCE > 0.0 and self.opt.netF == 'mlp_sample' and self.train_F:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # set train/eval modes
        if self.use_control and getattr(self.opt, 'freeze_G', True):
            self.netG.eval()
        else:
            self.netG.train()
        self.netE.train()
        self.netD.train()
        if hasattr(self, 'netC'):
            self.netC.train()
        if hasattr(self, 'netF'):
            self.netF.train() if self.train_F else self.netF.eval()

        compute_generator_update = (not self.opt.use_dmd) or (self.iteration % self.opt.dfake_gen_update_ratio == 0)

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update E
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        # update Guidance (DMD2: fake_unet + classifier)
        if self.opt.use_dmd:
            self.netGuidance.real_unet.freeze_parameters()
            self.set_requires_grad(self.netGuidance.fake_unet, True)
            if self.netGuidance.cls_pred_branch is not None:
                self.set_requires_grad(self.netGuidance.cls_pred_branch, True)
            self.optimizer_Guidance.zero_grad()
            self.loss_Guidance = self.compute_Guidance_loss()
            self.loss_Guidance.backward()
            self.optimizer_Guidance.step()

        # update G / ControlNet (respects DMD2 update ratio)
        if compute_generator_update:
            self.set_requires_grad(self.netD, False)
            self.set_requires_grad(self.netE, False)
            if self.opt.use_dmd:
                # fake_unet/cls must keep requires_grad=True for autograd.grad
                self.netGuidance.real_unet.freeze_parameters()
                self.set_requires_grad(self.netGuidance.fake_unet, True)
                if self.netGuidance.cls_pred_branch is not None:
                    self.set_requires_grad(self.netGuidance.cls_pred_branch, True)

            if self.optimizer_G is not None:
                self.optimizer_G.zero_grad()
            if self.use_control and self.optimizer_C is not None:
                self.optimizer_C.zero_grad()
            if self.opt.netF == 'mlp_sample' and self.train_F and hasattr(self, 'optimizer_F'):
                self.optimizer_F.zero_grad()

            self.loss_G = self.compute_G_loss()
            self.loss_G.backward()

            if self.optimizer_G is not None:
                self.optimizer_G.step()
            if self.use_control and self.optimizer_C is not None:
                self.optimizer_C.step()
            if self.opt.netF == 'mlp_sample' and self.train_F and hasattr(self, 'optimizer_F'):
                self.optimizer_F.step()
        else:
            self.loss_G = torch.tensor(0.0, device=self.device)
            if self.opt.use_dmd:
                self.loss_DM = getattr(self, 'loss_DM', torch.tensor(0.0, device=self.device))
                self.loss_G_CLS = getattr(self, 'loss_G_CLS', torch.tensor(0.0, device=self.device))

        # Increment iteration counter for DMD
        if self.opt.use_dmd:
            self.iteration += 1

    def set_input(self, input, input2=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # masks for ControlNet condition building
        self.mask_A = self._extract_mask(input, domain='A' if AtoB else 'B')
        self.mask_B = self._extract_mask(input, domain='B' if AtoB else 'A')
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)
            self.mask_A2 = self._extract_mask(input2, domain='A' if AtoB else 'B')
            self.mask_B2 = self._extract_mask(input2, domain='B' if AtoB else 'A')
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs = self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
        self.time_idx = time_idx
        self.timestep = times[time_idx]

        # build conditions from x0 for ControlNet
        if self.use_control:
            mask_A  = self._get_mask(self.real_A, self.mask_A)
            mask_A2 = self._get_mask(getattr(self, 'real_A2', self.real_A), getattr(self, 'mask_A2', None))
            mask_B  = self._get_mask(self.real_B, self.mask_B)
            cond_A_x0  = self._build_condition(self.real_A, mask_A)
            cond_A2_x0 = self._build_condition(getattr(self, 'real_A2', self.real_A), mask_A2)
            cond_B_x0  = self._build_condition(self.real_B, mask_B)
        else:
            mask_A = mask_A2 = mask_B = None
            cond_A_x0 = cond_A2_x0 = cond_B_x0 = None

        with torch.no_grad():
            self.netG.eval()
            if self.use_control:
                self.netC.eval()

            for t in range(self.time_idx.int().item()+1):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)

                # branch A
                Xt    = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx_t = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                z     = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
                ctrl  = self.netC(cond_A_x0, time_idx_t) if self.use_control else None
                Xt_1  = self.netG(Xt, time_idx_t, z, ctrl=ctrl)

                # branch A2
                Xt2   = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx_t2 = (t * torch.ones(size=[self.real_A2.shape[0]]).to(self.real_A.device)).long()
                z2    = torch.randn(size=[self.real_A2.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
                ctrl2 = self.netC(cond_A2_x0, time_idx_t2) if self.use_control else None
                Xt_12 = self.netG(Xt2, time_idx_t2, z2, ctrl=ctrl2)

                # branch B (identity)
                if self.opt.nce_idt:
                    XtB    = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                    time_idx_tb = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    zb     = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
                    ctrlb  = self.netC(cond_B_x0, time_idx_tb) if self.use_control else None
                    Xt_1B  = self.netG(XtB, time_idx_tb, zb, ctrl=ctrlb)

            if self.opt.nce_idt:
                self.XtB = XtB.detach()
            self.real_A_noisy  = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()

        # build condition for final step (optionally from xt)
        if self.use_control and self.opt.cond_ref == 'xt':
            cond_A  = self._build_condition(self.real_A_noisy, mask_A)
            cond_A2 = self._build_condition(self.real_A_noisy2, mask_A2)
            cond_B  = self._build_condition(self.XtB if self.opt.nce_idt else self.real_B, mask_B)
        else:
            cond_A, cond_A2, cond_B = cond_A_x0, cond_A2_x0, cond_B_x0

        z_in  = torch.randn(size=[2*bs, 4*self.opt.ngf]).to(self.real_A.device)
        z_in2 = torch.randn(size=[bs,   4*self.opt.ngf]).to(self.real_A.device)

        self.real  = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy

        if self.use_control:
            cond_main = torch.cat((cond_A, cond_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else cond_A
        else:
            cond_main = None

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real  = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])
                if self.use_control:
                    cond_main = torch.flip(cond_main, [3])

        ctrl_main = None
        if self.use_control:
            ctrl_main = self.netC(cond_main, self.time_idx)
            self._record_ctrl_norms(ctrl_main)
        else:
            self.ctrl_norm_down = self.ctrl_norm_mid = self.ctrl_norm_up = 0.0

        self.fake    = self.netG(self.realt, self.time_idx, z_in, ctrl=ctrl_main)
        ctrl_A2      = self.netC(cond_A2, self.time_idx) if self.use_control else None
        self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2, ctrl=ctrl_A2)
        self.fake_B  = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        if self.opt.phase == 'test':
            tau = self.opt.tau
            T   = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1), times])
            times = torch.tensor(times).float().cuda()
            self.times    = times
            bs            = self.real.size(0)
            time_idx      = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            self.time_idx = time_idx
            self.timestep = times[time_idx]
            with torch.no_grad():
                self.netG.eval()
                if self.use_control:
                    self.netC.eval()
                for t in range(self.opt.num_timesteps):
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx_t = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    z        = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
                    ctrl     = self.netC(cond_A_x0, time_idx_t) if self.use_control else None
                    Xt_1     = self.netG(Xt, time_idx_t, z, ctrl=ctrl)
                    setattr(self, "fake_"+str(t+1), Xt_1)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs  = self.real_A.size(0)
        fake = self.fake_B.detach()
        std  = torch.rand(size=[1]).item() * self.opt.std

        pred_fake = self.netD(fake, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real   = self.netD(self.real_B, self.time_idx)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):
        """Calculate SB energy estimator loss"""
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() + temp + temp**2
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

        total_loss = self.loss_Fake + self.opt.lambda_CLS * self.loss_Guidance_CLS

        self.fake_pred_noise_mean   = log_dict.get('faketrain_pred_noise_mean',   torch.tensor(0.0, device=self.device))
        self.fake_pred_noise_std    = log_dict.get('faketrain_pred_noise_std',    torch.tensor(0.0, device=self.device))
        self.fake_target_noise_mean = log_dict.get('faketrain_target_noise_mean', torch.tensor(0.0, device=self.device))
        self.fake_target_noise_std  = log_dict.get('faketrain_target_noise_std',  torch.tensor(0.0, device=self.device))
        self.fake_noise_mae         = log_dict.get('faketrain_noise_mae',         torch.tensor(0.0, device=self.device))

        return total_loss

    def compute_G_loss(self):
        bs  = self.real_A.size(0)
        tau = self.opt.tau
        fake = self.fake_B
        std  = torch.rand(size=[1]).item() * self.opt.std

        # GAN loss
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # SB loss
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            bs = self.opt.batch_size
            ET_XY = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB  = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY
            self.loss_SB += self.opt.tau*torch.mean((self.real_A_noisy-self.fake_B)**2)

        # NCE loss
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # flip-equivariance alignment for aux losses
        fake_aux = fake
        if getattr(self, 'flipped_for_equivariance', False):
            fake_aux = torch.flip(fake_aux, [3])

        # Frequency loss
        self.loss_freq = self.loss_freq_low = self.loss_freq_high = 0.0
        if getattr(self.opt, 'lambda_freq', 0.0) > 0.0:
            ref_img = self.real_A if self.opt.loss_ref == 'x0' else self.real_A_noisy
            freq_total, freq_low, freq_high = self._frequency_loss(fake_aux, ref_img, self.real_B, return_parts=True)
            self.loss_freq_low  = freq_low  * self.opt.lambda_freq
            self.loss_freq_high = freq_high * self.opt.lambda_freq
            self.loss_freq      = freq_total * self.opt.lambda_freq

        # Gram style loss
        self.loss_gram = 0.0
        if getattr(self.opt, 'lambda_gram', 0.0) > 0.0:
            if int(self.time_idx.item()) >= int(self.opt.gram_start_idx):
                self.loss_gram = self._gram_style_loss(fake_aux, self.real_B, self.time_idx) * self.opt.lambda_gram

        # Boundary consistency loss
        self.loss_bd = 0.0
        if getattr(self.opt, 'lambda_bd', 0.0) > 0.0:
            ref_img = self.real_A if self.opt.loss_ref == 'x0' else self.real_A_noisy
            maskA = self._get_mask(self.real_A, self.mask_A)
            self.loss_bd = self._boundary_consistency_loss(fake_aux, ref_img, maskA, self.opt.boundary_width) * self.opt.lambda_bd

        # DMD2 Distribution Matching Loss
        if self.opt.use_dmd:
            fake_for_dmd = self.fake_B if getattr(self.opt, 'dmd_update_ctrl', True) else self.fake_B.detach()
            loss_dict, _ = self.netGuidance(
                clean_images=fake_for_dmd,
                real_images=self.real_B,
                generator_turn=True,
                guidance_turn=False,
            )
            self.loss_DM = loss_dict['loss_dm']
            if self.opt.lambda_CLS > 0 and 'gen_cls_loss' in loss_dict:
                self.loss_G_CLS = loss_dict['gen_cls_loss']
            else:
                self.loss_G_CLS = torch.tensor(0.0, device=self.device)

            self.loss_G = (self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both
                           + self.loss_freq + self.loss_gram + self.loss_bd
                           + self.opt.lambda_DM * self.loss_DM + self.opt.lambda_CLS * self.loss_G_CLS)
        else:
            self.loss_G = (self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both
                           + self.loss_freq + self.loss_gram + self.loss_bd)

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z = torch.randn(size=[self.real_A.size(0), 4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.time_idx*0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # ------------------------------------------------------------------
    # Checkpoint load (skips missing files so net_C initializes fresh)
    # ------------------------------------------------------------------
    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            if self.opt.isTrain and self.opt.pretrained_name is not None:
                load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
            else:
                load_dir = self.save_dir
            load_path = os.path.join(load_dir, load_filename)
            if not os.path.isfile(load_path):
                print(f'[load_networks] {load_path} not found, skipping (random init).')
                continue
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print(f'[load_networks] loading {name} from {load_path}')
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            state_dict = {k: v for k, v in state_dict.items()
                          if not (k.endswith('.filt') or '.filt' in k)}
            net.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Helpers: pretrained loading
    # ------------------------------------------------------------------
    def _load_pretrained(self, net, ckpt_path, name='G'):
        if not ckpt_path:
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        # strip 'module.' prefix (DDP checkpoints)
        state_dict = {(k[len('module.'):] if k.startswith('module.') else k): v
                      for k, v in state_dict.items()}
        # drop filt buffers (shape may mismatch)
        state_dict = {k: v for k, v in state_dict.items()
                      if not (k.endswith('.filt') or '.filt' in k)}
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        is_filt = lambda ks: [k for k in ks if k.endswith('.filt') or '.filt' in k]
        missing_bad   = [k for k in missing   if k not in is_filt(missing)]
        unexpected_bad = [k for k in unexpected if k not in is_filt(unexpected)]
        if missing_bad or unexpected_bad:
            raise RuntimeError(
                f'[pretrained:{name}] Non-filt key mismatch!\n'
                f'  missing_bad={missing_bad[:20]}\n'
                f'  unexpected_bad={unexpected_bad[:20]}'
            )
        print(f'[pretrained:{name}] loaded from {ckpt_path}')

    # ------------------------------------------------------------------
    # Helpers: mask / condition
    # ------------------------------------------------------------------
    def _extract_mask(self, input_dict: dict, domain: str):
        key = getattr(self.opt, 'mask_key', '') or ''
        if key == '':
            return None
        if key in input_dict:
            return self._as_1ch_mask(input_dict[key].to(self.device))
        for k in [f'{domain}_mask', f'{domain}mask', 'mask', 'brain_mask', 'brainmask']:
            if k in input_dict:
                return self._as_1ch_mask(input_dict[k].to(self.device))
        return None

    @staticmethod
    def _as_1ch_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.size(1) != 1:
            mask = mask[:, :1]
        mask = mask.float()
        if mask.min() < 0:
            mask = (mask > 0).float()
        else:
            mask = mask.clamp(0.0, 1.0)
        return mask

    def _get_mask(self, img: torch.Tensor, provided_mask=None) -> torch.Tensor:
        if provided_mask is not None:
            return self._as_1ch_mask(provided_mask)
        thr = float(getattr(self.opt, 'mask_threshold', 0.01))
        return (img.abs().sum(dim=1, keepdim=True) > thr).float()

    def _build_condition(self, ref_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.opt.cond_use_lowpass:
            parts.append(self._gaussian_blur(ref_img, sigma=float(self.opt.low_sigma)))
        if self.opt.cond_use_grad:
            parts.append(self._sobel_grad_mag(ref_img))
        if self.opt.cond_use_boundary:
            parts.append(self._boundary_band(mask, width=int(self.opt.boundary_width)))
        return torch.cat(parts, dim=1)

    def _record_ctrl_norms(self, ctrl: dict):
        if not ctrl:
            self.ctrl_norm_down = self.ctrl_norm_mid = self.ctrl_norm_up = 0.0
            return
        down_vals, mid_vals, up_vals = [], [], []
        for name, t in ctrl.items():
            if not torch.is_tensor(t):
                continue
            v = t.detach().float().pow(2).mean().sqrt()
            if name.startswith('down'):
                down_vals.append(v)
            elif name.startswith('res'):
                mid_vals.append(v)
            elif name.startswith('up'):
                up_vals.append(v)
        z = torch.tensor(0.0, device=self.device)
        self.ctrl_norm_down = torch.stack(down_vals).mean() if down_vals else z
        self.ctrl_norm_mid  = torch.stack(mid_vals).mean()  if mid_vals  else z
        self.ctrl_norm_up   = torch.stack(up_vals).mean()   if up_vals   else z

    # ------------------------------------------------------------------
    # Helpers: image kernels / ops
    # ------------------------------------------------------------------
    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return x
        b, c, h, w = x.shape
        key = (c, float(sigma), x.device.type, str(x.dtype))
        if key not in self._gauss_cache:
            k = int(4.0 * sigma + 1.0)
            if k % 2 == 0:
                k += 1
            half = k // 2
            xs = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
            kernel1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
            kernel1d = kernel1d / kernel1d.sum()
            kx = kernel1d.view(1, 1, 1, k).repeat(c, 1, 1, 1)
            ky = kernel1d.view(1, 1, k, 1).repeat(c, 1, 1, 1)
            self._gauss_cache[key] = (kx, ky, half)
        kx, ky, pad = self._gauss_cache[key]
        x = F.conv2d(x, kx, padding=(0, pad), groups=c)
        x = F.conv2d(x, ky, padding=(pad, 0), groups=c)
        return x

    def _sobel_grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device.type, str(x.dtype))
        if key not in self._sobel_cache:
            kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 8.0
            ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 8.0
            self._sobel_cache[key] = (kx, ky)
        kx, ky = self._sobel_cache[key]
        gray = x.mean(dim=1, keepdim=True) if x.size(1) > 1 else x
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)

    @staticmethod
    def _boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
        if width <= 0:
            return torch.zeros_like(mask)
        k = 2 * width + 1
        dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=width)
        ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=width)
        return (dil - ero).clamp(0.0, 1.0)

    def _get_rfft_high_mask(self, h: int, w: int, cutoff: float, device) -> torch.Tensor:
        key = (h, w, float(cutoff), device.type)
        if key not in self._rfft_mask_cache:
            fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1)
            fx = torch.fft.rfftfreq(w, d=1.0, device=device).view(1, w // 2 + 1)
            self._rfft_mask_cache[key] = (torch.sqrt(fy ** 2 + fx ** 2) >= cutoff).float()
        return self._rfft_mask_cache[key]

    @staticmethod
    def _masked_l1(a, b, mask, eps=1e-8):
        if mask is None:
            return torch.mean(torch.abs(a - b))
        if mask.size(1) == 1 and a.size(1) > 1:
            mask = mask.repeat(1, a.size(1), 1, 1)
        return (torch.abs(a - b) * mask).sum() / (mask.sum() + eps)

    # ------------------------------------------------------------------
    # Extra losses (ControlNet)
    # ------------------------------------------------------------------
    def _frequency_loss(self, fake, ref, real_B, return_parts=False):
        sigma = float(self.opt.low_sigma)
        maskA = self._get_mask(ref, self.mask_A if hasattr(self, 'mask_A') else None)
        loss_low = self._masked_l1(self._gaussian_blur(fake, sigma), self._gaussian_blur(ref, sigma), maskA)

        b, c, h, w = fake.shape
        mag_fake = torch.abs(torch.fft.rfft2(fake, norm='ortho'))
        mag_real = torch.abs(torch.fft.rfft2(real_B, norm='ortho'))
        cutoff    = float(self.opt.freq_cutoff)
        high_mask = self._get_rfft_high_mask(h, w, cutoff, fake.device).view(1, 1, h, w // 2 + 1)
        loss_high = (torch.abs(mag_fake - mag_real) * high_mask).sum() / (high_mask.sum() * b * c + 1e-8)

        low_term  = float(self.opt.freq_w_low)  * loss_low
        high_term = float(self.opt.freq_w_high) * loss_high
        total = low_term + high_term
        if return_parts:
            return total, low_term, high_term
        return total

    @staticmethod
    def _gram_matrix(feat):
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / float(c * h * w)

    def _gram_style_loss(self, fake, real, time_idx):
        _, feats_f = self.netD(fake, time_idx, return_feats=True)
        with torch.no_grad():
            _, feats_r = self.netD(real, time_idx, return_feats=True)
        layers = [int(s.strip()) for s in str(self.opt.gram_layers).split(',') if s.strip()]
        loss, cnt = 0.0, 0
        for li in layers:
            if li < 0 or li >= len(feats_f) or li >= len(feats_r):
                continue
            loss += F.l1_loss(self._gram_matrix(feats_f[li]), self._gram_matrix(feats_r[li]))
            cnt += 1
        return loss / cnt if cnt > 0 else torch.tensor(0.0, device=fake.device)

    def _boundary_consistency_loss(self, fake, ref, mask, width):
        band  = self._boundary_band(mask, width=width)
        g_fake = self._sobel_grad_mag(fake)
        g_ref  = self._sobel_grad_mag(ref)
        return (torch.abs(g_fake - g_ref) * band).sum() / (band.sum() + 1e-8)
