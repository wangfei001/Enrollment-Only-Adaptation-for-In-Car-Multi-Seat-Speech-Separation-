#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .sodas_crnn import crnn
from .sodas_feature import FeatureExtractor, AngleFeature
from fast_transformers.attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask



class SharedLinearAttention(nn.Module):
    """Shared interface backed by the public fast-transformers LinearAttention."""
    def __init__(self, d_model):
        super().__init__()
        self.attn = LinearAttention(query_dimensions=d_model)

    def forward(self, q, k, v, m1=None, m2=None, m3=None, causal=False):
        # All call sites use FullMask. Bypass float32 mask multiplication so
        # the same linear-attention math can run fully in FP16.
        self.attn.feature_map.new_feature_map(q.device)
        q = self.attn.feature_map.forward_queries(q)
        k = self.attn.feature_map.forward_keys(k)
        kv = torch.einsum("nshd,nshm->nhmd", k, v)
        z = 1.0 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.attn.eps)
        return torch.einsum("nlhd,nhmd,nlh->nlhm", q, kv, z).contiguous()



class MISO1_MISO2(nn.Module):
    def __init__(self):
        super(MISO1_MISO2, self).__init__()

        self.doa_est = crnn()

        self.MISO2 = MISO2(
                use_dense = False,
                use_att = [True, True, True, True, False],
                use_rnn = [True, True, False, False, False],
                width = 64,
                num_layers = 4,
                dropout_rate = 0.4,
                causal_conf = False,
                )
        
    def forward(self, mixture_raw, mixture, do_eval = 0):
        assert do_eval == 0 or do_eval == 1, "Eval type should be 0 (training) or 1 (multi-channel beamforming) !"

        est_doa = self.doa_est(mixture_raw)    # b 2
        o_MISO2 = self.MISO2(mixture, est_doa)

        if do_eval == 0:
            return est_doa, o_MISO2
        else:
            return o_MISO2[:,:,0], o_MISO2


class PhaseHead(nn.Module):
    """Predict per microphone-pair, per-frequency phase correction from e_reg."""
    def __init__(self, dim=512, pairs=7, bins=257, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, pairs * bins),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb):
        return self.net(emb.float())


class DirectionWeightNet(nn.Module):
    """Frequency-dependent soft direction prior from registration embedding."""
    def __init__(self, dim=512, bins=72, rank=4, hidden=128, num_bins=257):
        super().__init__()
        self.bins = bins
        self.rank = rank
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, bins * rank),
        )
        self.bases = nn.Parameter(torch.randn(bins, rank, num_bins) * 0.02)
        self.bias = nn.Parameter(torch.zeros(bins, 1))

    def forward(self, emb):
        b = emb.shape[0]
        h = self.net(emb.float()).view(b, self.bins, self.rank)
        logits = torch.einsum('bkr,krf->bkf', h, self.bases.to(h.dtype))
        logits = logits + self.bias.to(h.dtype)[None]
        return torch.softmax(logits, dim=1)


class MISO2(nn.Module):
    def __init__(self,
                 use_dense,
                 use_att,
                 use_rnn,
                 num_channels=8,
                 num_spks=2,
                 frame_len=512,
                 frame_hop=128,
                 width=48,
                 num_layers=3,
                 dropout_rate=0.4,
                 causal_conf=False,
                 fdim = 16, # TODO
                 target_ref_only=False,
                 condition_mode='gt_doa',
                 reg_cond_dim=512,
                 num_doa_bins=72,
                 doa_weight_rank=4,
                 doa_weight_hidden=128,
                ):
        super(MISO2, self).__init__()

        self.use_dense = use_dense
        self.use_att = use_att
        self.use_rnn = use_rnn

        self.num_channels = num_channels
        self.num_spks = num_spks
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.width = width
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.causal_conf = causal_conf
        self.target_ref_only = target_ref_only
        self.condition_mode = condition_mode
        self.num_bins = self.frame_len // 2 + 1
        self.fdim = fdim

        self.extractor = FeatureExtractor(frame_len=self.frame_len, frame_hop=self.frame_hop, do_ipd=True, do_doa=True,
            ipd_index="1,0;2,0;3,0;4,0;5,0;6,0;7,0",
            ang_index="1,0;2,0;3,0;4,0;5,0;6,0;7,0")
        if self.condition_mode == 'reg_af':
            angles = torch.linspace(0.0, 2.0 * np.pi, num_doa_bins + 1)[:-1]
            with torch.no_grad():
                delay = self.extractor.ang_extractor._oracle_phase_delay(angles)
                delta = delay[:, self.extractor.ang_extractor.index_l] - \
                        delay[:, self.extractor.ang_extractor.index_r]
            self.register_buffer('reg_cos_delta', torch.cos(delta), persistent=False)
            self.register_buffer('reg_sin_delta', torch.sin(delta), persistent=False)
            self.dir_weight_net = DirectionWeightNet(
                dim=reg_cond_dim, bins=num_doa_bins, rank=doa_weight_rank,
                hidden=doa_weight_hidden, num_bins=self.num_bins,
            )
        else:
            self.dir_weight_net = None
        if self.condition_mode == 'reg_2d':
            self.reg_2d_net = nn.Sequential(
                nn.LayerNorm(reg_cond_dim), nn.Linear(reg_cond_dim, doa_weight_hidden),
                nn.GELU(), nn.Linear(doa_weight_hidden, 2),
            )
        else:
            self.reg_2d_net = None
        if self.condition_mode == 'reg_film':
            self.film_trunk = nn.Sequential(
                nn.LayerNorm(reg_cond_dim), nn.Linear(reg_cond_dim, doa_weight_hidden),
                nn.GELU(),
            )
            self.film_in = nn.Linear(doa_weight_hidden, 2 * self.width)
            self.film_blk = nn.Linear(doa_weight_hidden, 2 * self.width)
            nn.init.zeros_(self.film_in.weight); nn.init.zeros_(self.film_in.bias)
            nn.init.zeros_(self.film_blk.weight); nn.init.zeros_(self.film_blk.bias)
        else:
            self.film_trunk = self.film_in = self.film_blk = None
        if self.condition_mode == 'reg_phase':
            self.phase_head = PhaseHead(dim=reg_cond_dim, pairs=7,
                                        bins=self.num_bins, hidden=doa_weight_hidden)
        else:
            self.phase_head = None
        self.bn = nn.BatchNorm1d(self.num_bins * self.fdim)

        self.in_Conv = nn.Sequential(
                nn.Conv2d(in_channels=2*self.fdim, out_channels=self.width, kernel_size=(1, 1)),
                nn.LayerNorm(self.num_bins // 2 + 1),
                nn.PReLU(self.width),
                )

        self.in_Conv_att = nn.Sequential(nn.Conv2d(self.width, self.width // 2, kernel_size=(1, 1)), nn.PReLU())

        self.dualrnn_attention = nn.ModuleList()
        for i in range (self.num_layers):
            self.dualrnn_attention.append(DualRNN_Attention(dropout_rate=self.dropout_rate, d_model=self.width//2, use_att=self.use_att[i], use_rnn=self.use_rnn[i]))

        self.out_Conv1 = nn.Sequential(nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1)), nn.Tanh(),)
        self.out_Conv2 = nn.Sequential(nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1)), nn.Sigmoid(),)

        self.out_Conv_att = nn.Sequential(nn.Conv2d(self.width // 2, self.width, kernel_size=(1, 1)), nn.PReLU())

        out_mask_channels = (self.num_spks * 2) if self.target_ref_only else (self.num_channels * self.num_spks * 2)
        self.out_Conv = nn.ConvTranspose2d(in_channels=self.width//2, out_channels=out_mask_channels, kernel_size=(1, 1))

        if (self.use_dense):
            self.in_DenseBlock = DenseBlock(init_ch=self.width, g1=64, g2=self.width)
            self.out_DenseBlock = DenseBlock(init_ch=self.width, g1=64, g2=self.width)
        else:
            self.in_DenseBlock = None
            self.out_DenseBlock = None
            
    def half_reshape(self, x, inverse):
        if(inverse):
            x = torch.cat([x[:,0:x.shape[1]//2,],x[:,x.shape[1]//2:x.shape[1]//2*2,]],-1)
            x = x[...,:-1]
            return x
        else:
            x = F.pad(x, (0,1,0,0), 'replicate')
            x = torch.cat([x[:,:,:,0:x.shape[-1]//2],x[:,:,:,x.shape[-1]//2:x.shape[-1]//2*2]],1)
            return x


    def _reg_phase_features(self, mixture, cond):
        """Compute RegPhase spatial features in FP32; backbone stays AMP."""
        with torch.autocast(device_type="cuda", enabled=False):
            real = self.extractor.stft(mixture.float(), cplx=True)
            real_spec_src, imag_spec_src = real
            mag = torch.sqrt(real_spec_src.square() + imag_spec_src.square())
            pha = torch.atan2(imag_spec_src, real_spec_src)
            f = torch.clamp(mag[:, 0], min=1e-8)
            if self.extractor.log_mag:
                f = torch.log(f)
            if self.extractor.mvn_mag:
                f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-8)
            phase = torch.tanh(self.phase_head(cond.float())).view(
                pha.shape[0], 7, self.num_bins) * math.pi
            ipd = pha[:, self.extractor.ipd_extractor.index_l] - pha[:, self.extractor.ipd_extractor.index_r]
            af_reg = torch.cos(ipd - phase[..., None]).mean(1)
            spatial = torch.cat([self.extractor.ipd_extractor(pha), af_reg], dim=1)
            feat = torch.cat([f, spatial], 1)
            feat = feat.to(self.bn.weight.dtype)
            b, nf, t = feat.size()
            feat = self.bn(feat).to(mixture.dtype).contiguous().view(
                b, nf // self.num_bins, self.num_bins, t).permute(0, 1, 3, 2)
        return real_spec_src, imag_spec_src, feat

    def forward(self, mixture, doa):
        if self.condition_mode == 'reg_phase':
            real_spec_src, imag_spec_src, feat = self._reg_phase_features(mixture, doa)
        else:
            real_spec_src, imag_spec_src = self.extractor.stft(mixture, cplx=True)
            mag = torch.sqrt(real_spec_src.square() + imag_spec_src.square())
            pha = torch.atan2(imag_spec_src, real_spec_src)
            f = torch.clamp(mag[:, 0], min=1e-8)
            if self.extractor.log_mag:
                f = torch.log(f)
            if self.extractor.mvn_mag:
                f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-8)
            feature = [f]
            if self.condition_mode == 'reg_af':
                weights = self.dir_weight_net(doa).to(pha.dtype)
                ipd = pha[:, self.extractor.ipd_extractor.index_l] - pha[:, self.extractor.ipd_extractor.index_r]
                cos_d = self.reg_cos_delta.to(pha.dtype)
                sin_d = self.reg_sin_delta.to(pha.dtype)
                c = torch.einsum('bkf,kcf->bcf', weights, cos_d)
                s = torch.einsum('bkf,kcf->bcf', weights, sin_d)
                af_reg = (torch.cos(ipd) * c[..., None] + torch.sin(ipd) * s[..., None]).mean(1)
                spatial = torch.cat([self.extractor.ipd_extractor(pha), af_reg], dim=1)
                feature.append(spatial)
            elif self.condition_mode == 'reg_2d':
                doa_vec = F.normalize(self.reg_2d_net(doa.float()), dim=-1)
                doa_angle = torch.atan2(doa_vec[:,1], doa_vec[:,0])
                feature.append(self.extractor.compute_spatial(mixture, doa=doa_angle, pha=pha))
            elif self.condition_mode == 'reg_film':
                ipd = self.extractor.ipd_extractor(pha)
                af_zero = torch.zeros_like(f)
                feature.append(torch.cat([ipd, af_zero], dim=1))
            elif self.extractor.has_spatial:
                doa = torch.atan2(doa[:,1], doa[:,0])
                feature.append(self.extractor.compute_spatial(mixture, doa=doa, pha=pha))
            feat = torch.cat(feature, 1)
            feat = feat.to(self.bn.weight.dtype)
            B, nF, T = feat.size()
            feat = self.bn(feat).to(mixture.dtype).contiguous().view(
                B, nF // self.num_bins, self.num_bins, T).permute(0,1,3,2)
        x = self.half_reshape(feat, False)
 # [b 2fdim t f=129]

        x = self.in_Conv(x)  # [b w=64 t f]
        if self.condition_mode == 'reg_film':
            film = self.film_in(self.film_trunk(doa.float()))
            g, b = film.chunk(2, dim=1)
            x = x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        if (self.use_dense):
            x = self.in_DenseBlock(x)

        x = self.in_Conv_att(x) # [b w=32 t f]

        for i in range (self.num_layers):
            x = self.dualrnn_attention[i](x) # [b w t f]

        x = self.out_Conv_att(x) # [b w=64 t f]

        x = self.out_Conv1(x) * self.out_Conv2(x) # [b w=64 t f]

        if (self.use_dense):
            x = self.out_DenseBlock(x)
            
        x = self.half_reshape(x, True)  # [b w=32 t f=257]

        cmask_est = self.out_Conv(x).transpose(2,3) # [b 2xsxm f t]
        cmask_est = cmask_est.chunk(self.num_spks, dim=1) # [b 2m f t] * 2

        est_sig = []
        channels = 1 if self.target_ref_only else self.num_channels
        for id_spk in range (self.num_spks):
            for id_chan in range (channels):
                rmask_est = cmask_est[id_spk][:,0+2*id_chan] # [b f t]
                imask_est = cmask_est[id_spk][:,1+2*id_chan]

                real_spec_est = rmask_est * real_spec_src[:,id_chan] - imask_est * imag_spec_src[:,id_chan]
                imag_spec_est = rmask_est * imag_spec_src[:,id_chan] + imask_est * real_spec_src[:,id_chan]
                if self.condition_mode == 'reg_phase':
                    with torch.autocast(device_type="cuda", enabled=False):
                        est_sig.append(self.extractor.istft(
                            real_spec_est.float(), imag_spec_est.float(), cplx=True))
                else:
                    est_sig.append(self.extractor.istft(
                        real_spec_est, imag_spec_est, cplx=True)) # [b n] * m * s

        output = torch.stack(est_sig, 1) # [b mxs n]
        output = torch.stack(output.chunk(self.num_spks,1), 1) # [b s m n]
        output = torch.nn.functional.pad(output,[0,mixture.shape[-1]-output.shape[-1]])

        return output


class DualRNN_Attention(nn.Module):

    def __init__(self, dropout_rate, d_model, nhead=4, use_att=False, use_rnn=False):
        super(DualRNN_Attention,self).__init__()
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        self.nhead = nhead
        self.use_att = use_att
        self.use_rnn = use_rnn

        self.bn = nn.BatchNorm1d(1)

        if (self.use_att):
            self.shared_att1 = SharedLinearAttention(self.d_model)
            self.linear_att1 = nn.Linear(self.d_model, self.d_model*3)
            self.shared_att2 = SharedLinearAttention(self.d_model)
            self.linear_att2 = nn.Linear(self.d_model, self.d_model*3)
            self.ln_att = nn.LayerNorm(self.d_model)
        else:
            self.shared_att1 = None
            self.linear_att1 = None
            self.shared_att2 = None
            self.linear_att2 = None
            self.ln_att = None

        if (self.use_rnn):
            self.rnn1 = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model * 2, num_layers=1, bias=False, bidirectional=True, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model * 2, num_layers=1, bias=False, bidirectional=True, batch_first=True)
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.linear_rnn1 = nn.Linear(in_features = self.d_model * 4, out_features = self.d_model)
            self.linear_rnn2 = nn.Linear(in_features = self.d_model * 4, out_features = self.d_model)
            #self.relu = nn.ReLU()
            self.ln_rnn = nn.LayerNorm(self.d_model)
        else:
            self.rnn1 = None
            self.rnn2 = None
            self.dropout = None
            self.linear_rnn1 = None
            self.linear_rnn2 = None
            #self.relu = None
            self.ln_rnn = None

    def forward(self,x): # [b w t f=129]
        B, W, T, F = x.size()
        att_in1 = x.permute(0,2,3,1).contiguous().view(B*T, F, -1)

        if (not self.use_att):
            att_out1 = att_in1
        else:
            q, k, v = self.linear_att1(att_in1).view(B*T, F, self.nhead, -1).chunk(3,-1)
            m1, m2, m3 = FullMask(q.shape[1], k.shape[1], device=x.device),FullMask(q.shape[0], q.shape[1], device=x.device),FullMask(k.shape[0], k.shape[1], device=x.device)
            att_out1 = self.shared_att1(q, k, v, m1, m2, m3, causal=False).reshape(B*T, F, -1)
            att_out1 = self.ln_att((att_in1 + att_out1).to(self.ln_att.weight.dtype)).to(x.dtype)

        rnn_in1 = att_out1 # [bxt f w]

        if (not self.use_rnn):
            rnn_out1 = rnn_in1
        else:
            rnn_out1, _ = self.rnn1(rnn_in1)
            rnn_out1 = self.linear_rnn1(self.dropout(rnn_out1))
            rnn_out1 = self.ln_rnn((att_in1 + rnn_out1).to(self.ln_rnn.weight.dtype)).to(x.dtype)

        rnn_out1 = rnn_out1.view(B, T, F, -1).permute(0,3,1,2) # [b w t f]

        rnn_out1 = (self.bn(rnn_out1.reshape(B,1,-1).to(self.bn.weight.dtype))).reshape(*rnn_out1.shape).to(x.dtype) # [b w t f]
        rnn_out1 = rnn_out1 + x

        att_in2 = rnn_out1.permute(0,3,2,1).contiguous().view(B*F, T, -1)

        if (not self.use_att):
            att_out2 = att_in2
        else:
            q, k, v = self.linear_att2(att_in2).view(B*F, T, self.nhead, -1).chunk(3,-1)
            m1, m2, m3 = FullMask(q.shape[1], k.shape[1], device=x.device),FullMask(q.shape[0], q.shape[1], device=x.device),FullMask(k.shape[0], k.shape[1], device=x.device)
            att_out2 = self.shared_att2(q, k, v, m1, m2, m3, causal=False).reshape(B*F, T, -1)
            att_out2 = self.ln_att((att_in2 + att_out2).to(self.ln_att.weight.dtype)).to(x.dtype)

        rnn_in2 = att_out2 # [bxf t w]

        if (not self.use_rnn):
            rnn_out2 = rnn_in2
        else:
            rnn_out2, _ = self.rnn2(rnn_in2)
            rnn_out2 = self.linear_rnn2(self.dropout(rnn_out2))
            rnn_out2 = self.ln_rnn((att_in2 + rnn_out2).to(self.ln_rnn.weight.dtype)).to(x.dtype)

        rnn_out2 = rnn_out2.view(B, F, T, -1).permute(0,3,2,1) # [b w t f]

        rnn_out2 = (self.bn(rnn_out2.reshape(B,1,-1).to(self.bn.weight.dtype))).reshape(*rnn_out2.shape).to(x.dtype) # [b w t f]
        rnn_out2 = rnn_out2 + rnn_out1

        return rnn_out2
      
class DenseBlock(nn.Module):

    def __init__(self, init_ch, g1, g2):
        super(DenseBlock,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(init_ch, g1, kernel_size=(2,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(g1*2, g1, kernel_size=(2,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(g1*3, g1, kernel_size=(2,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1,affine=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(g1*4, g2, kernel_size=(2,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g2,affine=False)
        )

    def forward(self,x):   # x [b 64 t f]

        y0 = self.conv1(x)[:,:,:-1]

        y0_x = torch.cat((x,y0),dim=1)
        y1 = self.conv2(y0_x)[:,:,:-1]

        y1_0_x = torch.cat((x,y0,y1),dim=1)
        y2 = self.conv3(y1_0_x)[:,:,:-1]

        y2_1_0_x = torch.cat((x,y0,y1,y2),dim=1)
        y3 = self.conv4(y2_1_0_x)[:,:,:-1]

        return y3
      
