import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import opt
from models.modules import KAEM
from models.res2net import res2net50_v1b_26w_4s
from torchvision.transforms.functional import rgb_to_grayscale
from module.EGA import make_laplace_pyramid
from module.DMC_LPA import Muti_scale
from module.efds import EFDSA

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

    def forward(self, x):
        x = self.conv1(x) 
        
        x = self.conv2(x) 
       
        x = self.upsample(x) 
        return x


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )


class HeadUpdator(nn.Module):
    def __init__(self, in_channels=64, feat_channels=64, out_channels=None, conv_kernel_size=1):
        super(HeadUpdator, self).__init__()
        
        self.conv_kernel_size = conv_kernel_size

        # C == feat
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels if out_channels else in_channels
        # feat == in == out
        self.num_in = self.feat_channels
        self.num_out = self.feat_channels

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.pred_transform_layer = nn.Linear(self.in_channels, self.num_in + self.num_out)
        self.head_transform_layer = nn.Linear(self.in_channels, self.num_in + self.num_out, 1)

        self.pred_gate = nn.Linear(self.num_in, self.feat_channels, 1)
        self.head_gate = nn.Linear(self.num_in, self.feat_channels, 1)

        self.pred_norm_in = nn.LayerNorm(self.feat_channels)
        self.head_norm_in = nn.LayerNorm(self.feat_channels)
        self.pred_norm_out = nn.LayerNorm(self.feat_channels)
        self.head_norm_out = nn.LayerNorm(self.feat_channels)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = nn.LayerNorm(self.feat_channels)
        self.activation = nn.ReLU(inplace=True)

     
        self.dwconv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True, groups=1)


        self.inceptdw = InceptionDWConv2d(64)



    def forward(self, feat, head, pred):

        

        bs, num_classes = head.shape[:2] # bs:8 numclasses :1
        # C, H, W = feat.shape[-3:]

        pred = self.upsample(pred) 
        
        pred = torch.sigmoid(pred) 
        

        pred_minus = -1*(torch.sigmoid(pred)) + 1
        pred_add = pred.mul(pred_minus) + pred
        pred_conv = self.dwconv(pred_add)
        pred = pred + pred_conv
        feat = self.inceptdw(feat)
    
      
                    


        """
        Head feature assemble 
        - use prediction to assemble head-aware feature
        """
   
     
        pred_expanded = pred.unsqueeze(2)  # (B, N, 1, H, W)
        feat_expanded = feat.unsqueeze(1)  # (B, 1, C, H, W)
        pred_fft = torch.fft.rfft2(pred_expanded.float())
        feat_fft = torch.fft.rfft2(feat_expanded.float())
     

     
        mult = pred_fft * feat_fft   # (B, N, C, H, W)
        
        assemble_feat = torch.fft.irfft2(mult)
      

       
        assemble_feat = assemble_feat.sum(dim=[3, 4])   # (B, N, C)
        
        

        # [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        head = head.reshape(bs, num_classes, self.in_channels, -1).permute(0, 1, 3, 2) # torch.Size([8, 1, 1, 64])
        
        """
        Update head
        - assemble_feat, head -> linear transform -> pred_feat, head_feat
        - both split into two parts: xxx_in & xxx_out
        - gate_feat = head_feat_in * pred_feat_in
        - gate_feat -> linear transform -> pred_gate, head_gate
        - update_head = pred_gate * pred_feat_out + head_gate * head_feat_out
        """
        # [B, N, C] -> [B*N, C]
        assemble_feat = assemble_feat.reshape(-1, self.in_channels) # torch.Size([8, 64])
        bs_num = assemble_feat.size(0) # bs_num :8

        # [B*N, C] -> [B*N, in+out]
        pred_feat = self.pred_transform_layer(assemble_feat) 
        
        # [B*N, in]
        pred_feat_in = pred_feat[:, :self.num_in].view(-1, self.feat_channels) 
        # [B*N, out]
        pred_feat_out = pred_feat[:, -self.num_out:].view(-1, self.feat_channels) 

        # [B, N, K*K, C] -> [B*N, K*K, C] -> [B*N, K*K, in+out]
        head_feat = self.head_transform_layer(
            head.reshape(bs_num, -1, self.in_channels)) # torch.Size([8, 1, 128])

        # [B*N, K*K, in]
        head_feat_in = head_feat[..., :self.num_in] # torch.Size([8, 1, 128])
        # [B*N, K*K, out]
        head_feat_out = head_feat[..., -self.num_out:] # torch.Size([8, 1, 64])

        # [B*N, K*K, in] * [B*N, 1, in] -> [B*N, K*K, in]
        gate_feat = head_feat_in * pred_feat_in.unsqueeze(-2) # torch.Size([8, 1, 64]) 

        # [B*N, K*K, feat]
        head_gate = self.head_norm_in(self.head_gate(gate_feat)) 
        pred_gate = self.pred_norm_in(self.pred_gate(gate_feat)) 


        head_gate = torch.sigmoid(head_gate) # torch.Size([8, 1, 64])
        pred_gate = torch.sigmoid(pred_gate) # torch.Size([8, 1, 64])

        # [B*N, K*K, out]
        head_feat_out = self.head_norm_out(head_feat_out) # torch.Size([8, 1, 64])
        # [B*N, out]
        pred_feat_out = self.pred_norm_out(pred_feat_out) # torch.Size([8, 64])

        # [B*N, K*K, feat] or [B*N, K*K, C]
        update_head = pred_gate * pred_feat_out.unsqueeze(-2) + head_gate * head_feat_out # torch.Size([8, 1, 64]) # update_head对应论文中的Ki

        update_head = self.fc_layer(update_head) # torch.Size([8, 1, 64])
        update_head = self.fc_norm(update_head) # torch.Size([8, 1, 64])
        update_head = self.activation(update_head) # torch.Size([8, 1, 64])

        # [B*N, K*K, C] -> [B, N, K*K, C]
        update_head = update_head.reshape(bs, num_classes, -1, self.feat_channels) # torch.Size([8, 1, 1, 64])
        # [B, N, K*K, C] -> [B, N, C, K*K] -> [B, N, C, K, K]
        update_head = update_head.permute(0, 1, 3, 2).reshape(bs, num_classes, self.feat_channels, self.conv_kernel_size, self.conv_kernel_size) # torch.Size([8, 1, 64, 1, 1])

       

        return update_head


class EGMFA(nn.Module):
    def __init__(self, num_classes=1, unified_channels=64, conv_kernel_size=1):
        super(EGMFA, self).__init__()
        self.num_classes = num_classes
        self.conv_kernel_size = conv_kernel_size
        self.unified_channels = unified_channels

        res2net = res2net50_v1b_26w_4s(pretrained=True)
        
        # Encoder
        self.encoder1_conv = res2net.conv1
        self.encoder1_bn = res2net.bn1
        self.encoder1_relu = res2net.relu
        self.maxpool = res2net.maxpool
        self.encoder2 = res2net.layer1
        self.encoder3 = res2net.layer2
        self.encoder4 = res2net.layer3
        self.encoder5 = res2net.layer4

        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 512, 1)
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512+256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256+128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128+64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64+64, out_channels=64)

     
        self.gobal_average_pool = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        #self.gobal_average_pool = nn.AdaptiveAvgPool2d(1)
        self.generate_head = nn.Linear(512, self.num_classes*self.unified_channels*self.conv_kernel_size*self.conv_kernel_size)

       
        self.headUpdators = nn.ModuleList()
        for i in range(4):
            self.headUpdators.append(HeadUpdator())

        self.unify1 = nn.Conv2d(64, 64, 1)
        self.unify2 = nn.Conv2d(64, 64, 1)
        self.unify3 = nn.Conv2d(128, 64, 1)
        self.unify4 = nn.Conv2d(256, 64, 1)
        self.unify5 = nn.Conv2d(512, 64, 1)

    


       
        self.KAEM1 = KAEM(dim=64)
        self.KAEM2 = KAEM(dim=128)
        self.KAEM3 = KAEM(dim=256)
        self.KAEM4 = KAEM(dim=512)
   

        self.decoderList = nn.ModuleList([self.decoder4, self.decoder3, self.decoder2, self.decoder1])
        self.unifyList = nn.ModuleList([self.unify4, self.unify3, self.unify2, self.unify1])
        
        self.KAEMList = nn.ModuleList([self.KAEM4, self.KAEM3, self.KAEM2, self.KAEM1])
       
    

     
        up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        mutil_channel = [64, 64, 128, 256, 512]
        self.Muti_scale = Muti_scale(mutil_channel, up_kwargs=up_kwargs)
    
      
        self.efds1 = EFDSA(dim=64,resolution = opt.img_size//2)
        self.efds2 = EFDSA(dim=64,resolution = opt.img_size//4)
        self.efds3 = EFDSA(dim=128,resolution = opt.img_size//8)
        self.efds4 = EFDSA(dim=256,resolution = opt.img_size//16)
        self.efds5 = EFDSA(dim=512,resolution = opt.img_size//32)

    def forward(self, x):

        grayscale_img = rgb_to_grayscale(x)
  
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

      
       
      
        bs = x.shape[0] 
        e1_ = self.encoder1_conv(x)  
        e1_ = self.encoder1_bn(e1_)  
        e1_ = self.encoder1_relu(e1_) 
        e1_pool_ = self.maxpool(e1_)  
        e2_ = self.encoder2(e1_pool_) 
        e3_ = self.encoder3(e2_)      
        e4_ = self.encoder4(e3_)      
        e5_ = self.encoder5(e4_)      
        
        e1 = e1_                
        e2 = self.reduce2(e2_)  
        e3 = self.reduce3(e3_)  
        e4 = self.reduce4(e4_)  
        e5 = self.reduce5(e5_)  


        e1_dmc, e2_dmc, e3_dmc, e4_dmc, e5_dmc = self.Muti_scale(e1, e2, e3, e4, e5, edge_feature) #整理为一个模块

     

        y1 = self.efds1(e1_dmc)
        y2 = self.efds2(e2_dmc)
        y3 = self.efds3(e3_dmc)
        y4 = self.efds4(e4_dmc)
        y5 = self.efds5(e5_dmc)

    
        esa_out = [y4,y3,y2,y1]


        #e5 = self.esa5(e5)
        e5 = y5
        d5 = self.decoder5(e5)      # H/16*W/16*512 torch.Size([8, 512, 14, 14])
        
        feat5 = self.unify5(d5)     # torch.Size([8, 64, 14, 14])

        decoder_out = [d5]
        encoder_out = [e5, e4, e3, e2, e1]

     
        # [B, 512, 1, 1] -> [B, 512]
        gobal_context = self.gobal_average_pool(e5)   # torch.Size([8, 512, 1, 1])

        gobal_context = gobal_context.reshape(bs, -1) # torch.Size([8, 512])
        
        # [B, N*C*K*K] -> [B, N, C, K, K]
        head = self.generate_head(gobal_context)      # torch.Size([8, 64])

        head = head.reshape(bs, self.num_classes, self.unified_channels, self.conv_kernel_size, self.conv_kernel_size)  # torch.Size([8, 1, 64, 1, 1])
        
        pred = []
        for t in range(bs):
            pred.append(F.conv2d(
                feat5[t:t+1],  # feat5: torch.Size([8, 64, 14, 14])
                head[t],
                padding=int(self.conv_kernel_size // 2)))
        pred = torch.cat(pred, dim=0)
        H, W = feat5.shape[-2:]
        # [B, N, H, W]
        pred = pred.reshape(bs, self.num_classes, H, W)  # torch.Size([8, 1, 112, 112])
        stage_out = [pred]

        # feat size: [B, C, H, W]
        # feats = [feat4, feat3, feat2, feat1]
        feats = []



        for i in range(4):
            

           
            KAEM_out = self.KAEMList[i](decoder_out[-1], stage_out[-1])
            comb = torch.cat([KAEM_out, esa_out[i]], dim=1)

           
            
            d = self.decoderList[i](comb)
            decoder_out.append(d)
            
            feat = self.unifyList[i](d)
            feats.append(feat)

            head = self.headUpdators[i](feats[i], head, pred)  #.Size([8, 64, 28, 28])
            pred = []

            for j in range(bs):
                pred.append(F.conv2d(
                    feats[i][j:j+1],
                    head[j],
                    padding=int(self.conv_kernel_size // 2)))
            pred = torch.cat(pred, dim=0)
            H, W = feats[i].shape[-2:]
            pred = pred.reshape(bs, self.num_classes, H, W)
            stage_out.append(pred)
            
        stage_out.reverse()
        #return stage_out[0], stage_out[1], stage_out[2], stage_out[3], stage_out[4]
        return torch.sigmoid(stage_out[0]), torch.sigmoid(stage_out[1]), torch.sigmoid(stage_out[2]), \
               torch.sigmoid(stage_out[3]), torch.sigmoid(stage_out[4])
    