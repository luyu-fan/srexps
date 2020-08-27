"""
核心文件: 为了压缩文件夹组织规模，模块包括了数据处理，模型定义与训练等所有核心代码
"""
import torch,os,random,time,math
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import models, transforms
from torchvision.utils import make_grid

# region Utils
def data_statistic(lr_root,hr_root):
    r"""
    统计图像尺寸
    Args:
        lr_root: lr根目录
        hr_root: hr根目录
    """
    lr_imgs = os.listdir(lr_root)
    hr_imgs = os.listdir(hr_root)
    lr_size_statistic = [[0,0]] * len(lr_imgs)
    hr_size_statistic = [[0,0]] * len(hr_imgs)
    for i,lr_img in enumerate(lr_imgs):
        path = os.path.join(lr_root,lr_img)
        print("Processing: ",path)
        img = Image.open(path)
        lr_size_statistic[i] = img.size
    for i,hr_img in enumerate(hr_imgs):
        path = os.path.join(hr_root,hr_img)
        print("Processing: ",path)
        img = Image.open(path)
        hr_size_statistic[i] = img.size
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize = (6,6),dpi = 120)
    sns.scatterplot(x = [item[0] for item in lr_size_statistic],y = [item[1] for item in lr_size_statistic])
    sns.scatterplot(x = [item[0] for item in hr_size_statistic],y = [item[1] for item in hr_size_statistic])
    plt.show()
# endregion

# region Transforms
class CropImgPairs(object):
    
    def __init__(self):
        r"""
        对LR和HR对进行裁剪，得到Patch对
        """
        super(CropImgPairs,self).__init__()

    def __call__(self,lr_img,hr_img,corp_size,scale):
        r"""
        裁剪图像
        """
        lr_w_crop_pos = random.randint(0,lr_img.size[0]-corp_size[0])
        lr_h_crop_pos= random.randint(0,lr_img.size[1]-corp_size[1])
        hr_w_crop_pos = lr_w_crop_pos * scale
        hr_h_crop_posl = lr_h_crop_pos * scale
        lr_img = lr_img.crop((lr_w_crop_pos,lr_h_crop_pos,lr_w_crop_pos + corp_size[0],lr_h_crop_pos + corp_size[1]))
        hr_img = hr_img.crop((hr_w_crop_pos,hr_h_crop_posl,hr_w_crop_pos + corp_size[0] * scale,hr_h_crop_posl + corp_size[1] * scale))
        return lr_img,hr_img

class HorizontalFlip(object):
    def __init__(self,p = 0.5):
        r"""
        水平翻转
        """
        super(HorizontalFlip,self).__init__()
        self._trans = transforms.RandomHorizontalFlip(p=1)
        self.p = p

    def __call__(self,lr_img,hr_img):
        r"""
        翻转图像
        """
        if random.random() <= self.p:
            lr_img = self._trans(lr_img)
            hr_img = self._trans(hr_img)
        return lr_img,hr_img

class VerticalFlip(object):
    def __init__(self,p = 0.5):
        r"""
        竖直翻转
        Args:
            p: 概率
        """
        super(VerticalFlip,self).__init__()
        self._trans = transforms.RandomVerticalFlip(p=1)
        self.p = p

    def __call__(self,lr_img,hr_img):
        r"""
        翻转图像
        """
        if random.random() <= self.p:
            lr_img = self._trans(lr_img)
            hr_img = self._trans(hr_img)
        return lr_img,hr_img

class Rotation(object):
    def __init__(self,p = 0.5):
        r"""
        旋转图像
        """
        super(Rotation,self).__init__()
        self.p = p

    def __call__(self,lr_img,hr_img,degree):
        r"""
        旋转图像
        """
        if random.random() <= self.p: 
            lr_img = lr_img.rotate(degree)
            hr_img = hr_img.rotate(degree)
        return lr_img,hr_img
# endregion

# region Datasets
class CompetitionSRDataset(Dataset):

    def __init__(self,lr_root,hr_root,crop_size,sr_scale,mode = "train"):
        r"""
        有监督形式的数据集
        Args:
            lr_root: lr根目录
            hr_root: hr根目录
            crop_size: 图像裁剪
            sr_scale: SR倍数
            mode: 模式
        """
        self.lr_root = lr_root
        self.hr_root = hr_root 
        self.crop_size = crop_size
        self.sr_scale = sr_scale
        # transforms
        self.crop_transform = CropImgPairs()
        self.horizontal_transform = HorizontalFlip()
        self.vertical_transform = VerticalFlip()
        # self.rotatiom_transform = Rotation()
        self.totensor_transform = transforms.ToTensor()
        # mode
        self.mode = mode
    
    def prepare(self):
        r"""
        图像列表以及数据集划分,假设LR数据集和HR数据集的图像名称是一样的
        """
        self.lr_imgs = os.listdir(self.lr_root)
        if self.mode != "test":
            self.hr_imgs = os.listdir(self.hr_root)
            self.imgs = [(img,img) for img in self.lr_imgs if img in self.hr_imgs]
            if self.mode == "train":
                self.imgs = self.imgs[:int(len(self.imgs) * 0.8)]
            elif self.mode == "val":
                self.imgs = self.imgs[int(len(self.imgs) * 0.8):]
        else:
            self.imgs = self.lr_imgs
        
    def __len__(self):
        r"""
        数据集长度
        """
        return len(self.imgs)
        
    def __getitem__(self,index):
        r"""
        获得LR与HR对
        """
        if self.mode == "test":
            lr_img_path = os.path.join(self.lr_root,self.imgs[index])
            img = Image.open(lr_img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return self.totensor_transform(img)
        else:
            lr_img_path = os.path.join(self.lr_root,self.imgs[index][0])
            hr_img_path = os.path.join(self.hr_root,self.imgs[index][1])
            lr_img = Image.open(lr_img_path)
            hr_img = Image.open(hr_img_path)
            if lr_img.mode != "RGB":
                lr_img = lr_img.convert("RGB")
            if hr_img.mode != "RGB":
                hr_img = hr_img.convert("RGB")
            lr_img,hr_img = self.crop_transform(lr_img,hr_img,self.crop_size,self.sr_scale)
            lr_img,hr_img = self.horizontal_transform(lr_img,hr_img)
            lr_img,hr_img = self.vertical_transform(lr_img,hr_img)
            # lr_img,hr_img = self.rotatiom_transform(lr_img,hr_img,0.5 * math.pi)
            return self.totensor_transform(lr_img),self.totensor_transform(hr_img)

class DIV2KDataset(CompetitionSRDataset):
    
    def __init__(self,lr_root,hr_root,crop_size,sr_scale,mode = "train"):
        r"""
        针对DIV2k的处理数据集
        """
        super(DIV2KDataset,self).__init__(lr_root,hr_root,crop_size,sr_scale,mode)

    def prepare(self):
        r"""
        针对DIV2k数据集划分
        """
        lr_imgs = os.listdir(self.lr_root)
        if self.mode != "test":
            hr_imgs = os.listdir(self.hr_root)
            self.imgs = []
            for img in hr_imgs:
                img_splited = img.split(".")
                if img_splited[0]+"x"+str(self.sr_scale)+"."+img_splited[1] in lr_imgs:
                    self.imgs.append((img_splited[0]+"x"+str(self.sr_scale)+"."+img_splited[1],img))
            if self.mode == "train":
                self.imgs = self.imgs[:int(len(self.imgs) * 0.8)]
            elif self.mode == "val":
                self.imgs = self.imgs[int(len(self.imgs) * 0.8):]
        else:
            self.imgs = lr_imgs
# endregion

class Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias,act = None):
        r"""
        带有激活的卷积层,SAME模式
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            bias: 是否进行偏置项
            act: 激活
        """
        super(Conv2d,self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias = bias)
        
        self.act = None
        if act is not None:
            if act == "PReLU":
                self.act = nn.PReLU()
            elif act == "ReLU":
                self.act = nn.ReLU()
            elif act == "LeakyReLU":
                self.act = nn.LeakyReLU(negative_slope = 0.2)
            else:
                raise ValueError

    def forward(self,x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

class Conv2dTranspose(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias,act = None):
        r"""
        带有激活的转置卷积层，SAME模式
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            kernel_size: 卷积核大小
            stride: 步长，即缩放倍数
            bias: 是否进行偏置项
            act: 激活
        """
        super(Conv2dTranspose,self).__init__()

        self.convtrans = nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) - (kernel_size - 1) // 2,
            stride = stride,
            output_padding = stride - 1,
            bias = bias
        )

        self.act = None
        if act is not None:
            if act == "PReLU":
                self.act = nn.PReLU()
            elif act == "ReLU":
                self.act = nn.ReLU()
            elif act == "LeakyReLU":
                self.act = nn.LeakyReLU(negative_slope = 0.2)
            else:
                raise ValueError

    def forward(self,x):
        x = self.convtrans(x)
        if self.act is not None:
            x = self.act(x)
        return x

class GaussianModel(nn.Module):
    def __init__(self,img_channels,kernel_size,sigma):
        r"""
        对图像进行高斯滤波:得到高频和低频部分
        """
        super(GaussianModel,self).__init__()
        self.gaussian_conv = nn.Conv2d(img_channels,img_channels,kernel_size,2,padding=(kernel_size - 1)//2,groups = img_channels,bias = False)
        self.gaussian_deconv = nn.ConvTranspose2d(img_channels,img_channels,kernel_size,2,padding = (kernel_size - 1) - (kernel_size - 1)//2,groups = img_channels,output_padding = 1,bias = False)
        self.gaussian_conv.weight = nn.Parameter(self.get_gaussian_kernel(kernel_size,sigma,img_channels),requires_grad = False)
        self.gaussian_deconv.weight = nn.Parameter(4.0 * self.get_gaussian_kernel(kernel_size,sigma,img_channels),requires_grad = False)

    def get_gaussian_kernel(self,kernel_size,sigma,in_channels):
        r"""
        生成高斯卷积核
        Args:
            kernel_size: 高斯核大小
            sigma: 方差
            in_channels: 有多少个通道
        """
        kernel = torch.tensor([math.exp((-(i - kernel_size // 2) ** 2) / (2 * sigma ** 2)) for i in range(kernel_size)],dtype = torch.float32)
        kernel = kernel.unsqueeze(0)
        kernel = torch.matmul(kernel.t(),kernel).float()
        kernel /= kernel.sum()     # so (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) could be removed in kernel func, and whatever normalizing before or later is ok.
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(size=(in_channels,1,kernel_size,kernel_size)).contiguous()
        return kernel

    def forward(self, x):
        gaussian_img = self.gaussian_deconv(self.gaussian_conv(x))
        lap_img = x - gaussian_img
        return gaussian_img,lap_img

class ResidualModel(nn.Module):
    def __init__(self,channels,kernel_size,bias,act,residual_ratio = 0.1,res_del = False):
        r"""
        简化残差块
        """
        super(ResidualModel,self).__init__()

        self.residual_ratio = residual_ratio
        self.res_del = res_del

        self.conv1 = Conv2d(channels,channels,kernel_size,1,(kernel_size - 1) // 2,bias,act)
        if res_del:
            self.conv2 = Conv2d(channels,channels,kernel_size,1,(kernel_size - 1) // 2,bias,act)
        else:
            self.conv2 = Conv2d(channels,channels,kernel_size,1,(kernel_size - 1) // 2,bias,None)
        

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        output = x if self.res_del else (self.residual_ratio * input + x)
        return output

class DLAPSR(nn.Module):
    def __init__(self,scale_factor = 2):
        
        super(DLAPSR, self).__init__()

        img_channels=3
        feat_num=13
        m=3
        pixelshuffle = False
        act = "PReLU"

        # 特征抽取
        self.feat_extract = Conv2d(img_channels, feat_num, kernel_size = 3, stride = 1, padding = 1, bias = True,act = act)
    
        # 低频处理分支
        lf_branch = [ResidualModel(channels = feat_num, kernel_size=3, bias = True, act = act) for _ in range(m)]
        lf_branch.append(UpSampler(scale_factor,feat_num,img_channels,kernel_size=3,bias =True,act = act,use_pixelshuffle = pixelshuffle))


        # 高频处理分支
        hf_branch = [ResidualModel(channels = feat_num, kernel_size=3, bias = True, act = act, res_del = True) for _ in range(m)]
        hf_branch.append(UpSampler(scale_factor,feat_num,img_channels,kernel_size=3,bias =True,act = act,use_pixelshuffle = pixelshuffle))

        self.lf_branch = nn.Sequential(*lf_branch)
        self.hf_branch = nn.Sequential(*hf_branch)

    def forward(self,x):
        x = self.feat_extract(x)
        lf_img = self.lf_branch(x).clamp_(0.0,1.0)
        hf_img = self.hf_branch(x).clamp_(0.0,1.0)
        out_img = (lf_img + hf_img).clamp_(0.0,1.0)
        return out_img,lf_img,hf_img

class UpSampler(nn.Module):
    def __init__(self,sr_scale,in_channels,out_channels,kernel_size,bias,act,use_pixelshuffle = True):
        r"""
        使用转置卷积进行上采样部分
        Args:
            sr_scale: 上采样尺寸
            conv_num: 用于细节修正的卷积数目
            channels: 通道数
            kernel_size: 卷积核
            bias: 偏置
            act: 激活
        """
        super(UpSampler,self).__init__()

        if use_pixelshuffle:
            up_conv = Conv2d(in_channels,out_channels * (sr_scale ** 2),kernel_size,1,(kernel_size - 1) // 2,bias,act)
            pixel_shuffle = nn.PixelShuffle(sr_scale)
            self.upsample = nn.Sequential(up_conv,pixel_shuffle)
        else:
            self.upsample = Conv2dTranspose(in_channels,out_channels,kernel_size,sr_scale,bias,act)
    
    def forward(self, x):
        x = self.upsample(x)
        return x

def weights_init(model):
    r"""
    对网络权重进行初始化
    """
    if isinstance(model,(nn.Conv2d,nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(model.weight)
        nn.init.constant_(model.bias,0.0)

def train_dlapsr(args=None):
    
    # Configurtion
    crop_size = (64,64)
    sr_scale = 2
    coarse_train_epoch_num = 2400
    finetune_train_epoch_num = 600
    batch_size = 32
    workers_num = 2

    tensor2img = transforms.ToPILImage()

    # Dataset
    # hr_root = r"/workspace/sr/agora_sr/train/HR"
    # lr_root = r"/workspace/sr/agora_sr/train/LR"
    hr_root = r"D:\MasterDegree\Datasets\SR\agora_sr_2020.7.28_train\agora_sr_2020.7.28_train\train\HR"
    lr_root = r"D:\MasterDegree\Datasets\SR\agora_sr_2020.7.28_train\agora_sr_2020.7.28_train\train\LR"
    trainset = CompetitionSRDataset(lr_root,hr_root,crop_size,sr_scale,mode = "train")
    
    trainset.prepare()
    trainloader = DataLoader(trainset,batch_size = batch_size,shuffle = True,num_workers = workers_num, drop_last = True)

    # Seed
    torch.manual_seed(688898264246898)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = DLAPSR()
    model = model.to(device)
    model.apply(weights_init)

    gaussian_model = GaussianModel(3,3,4)
    gaussian_model = gaussian_model.to(device)
    gaussian_model = gaussian_model.eval()

    # Criterion
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(),lr = 1e-4,betas = (0.9,0.999),eps = 1e-8)

    # Scheduler
    # approximate every 50,000 iterations.
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size = 1000, gamma = 0.5)

    # Running
    print("-" * 120)
    print("Model:\n",model)

    # from thop import profile
    # print("-" * 120)
    # flops,params = profile(model,inputs = (torch.rand(size=(1,3,360,240)).to(device),))
    # print("Flops:",flops,"Training Maybe:",flops / 1e9)
    # print("Params:",params)

    # region Training
    print("-" * 120)
    print("---> Start training:")
    total_step = 0
    for epoch in range(coarse_train_epoch_num + finetune_train_epoch_num + 1):
        # criterion
        if epoch <= coarse_train_epoch_num:
            criterion = criterion1
        else:
            criterion = criterion2
        
        # statistic
        total_loss = 0
        start = time.time()
        for i,data in enumerate(trainloader):
            # iter
            total_step += 1
            # clean grad.
            optimizer.zero_grad()

            # input
            lr_tensor,hr_tensor = data
            lr_tensor = lr_tensor.to(device)
            hr_gt_tensor = hr_tensor.to(device)

            # output
            hr_tensor,lf_tensor,hf_tensor = model(lr_tensor)
            # gaussian model
            gaussian_img,laps_img = gaussian_model(hr_gt_tensor)

            # backward
            _g_loss = criterion(hr_tensor,hr_gt_tensor)
            _lf_loss = criterion(lf_tensor,gaussian_img)
            _hf_loss = criterion(hf_tensor,laps_img)
            loss = 0.1 * _g_loss + _lf_loss + 10 *_hf_loss

            loss.backward()
            total_loss += loss.item()

            # update
            optimizer.step()
            if total_step % 10 == 0 and total_step != 0:
                end = time.time()
                print("Epoch: {}/{}, Iter:{}/{}, Avg Total Loss:{}, Sample Details:{} / {} / {}, Time Mean Cost: {} s / iter".format(
                    epoch,
                    coarse_train_epoch_num+finetune_train_epoch_num, 
                    i,len(trainloader),
                    total_loss / 10,
                    0.1 * _g_loss.item(),
                    _lf_loss.item(),
                    10.0 * _hf_loss.item(),
                    (end - start) / 10)
                    )
                total_loss = 0
                start = time.time()

            # display
            if epoch % 50 == 0 and i % 20 == 0:
                outputs = torch.cat((hr_gt_tensor,hr_tensor,gaussian_img,lf_tensor,laps_img,hf_tensor),dim = 0)
                outputs_pathces = make_grid(outputs.cpu(),nrow = 16,padding = 4)
                record_img = tensor2img(outputs_pathces)
                record_img.save("./record_dlapsr_%d_%d.jpg" % (epoch,i))

        scheduler.step()
        if epoch % 50 == 0:
            torch.save(model.state_dict(),"./dlapsr_%d.pth" % epoch)
    torch.save(model.state_dict(),"./dlapsr_final.pth")
    # endregion

if __name__ == "__main__":
    
    # training
    train_dlapsr()

