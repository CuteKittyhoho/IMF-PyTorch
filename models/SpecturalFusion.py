import torch
import torch.nn as nn
#这段代码定义了一个名为 FeedForward 的PyTorch模块（Module），
#用于实现Transformer模型中的前馈神经网络（FeedForward Network）
#d_model：输入和输出的维度大小；d_ff：前馈神经网络中间层的维度大小；dropout：Dropout层的丢弃概率，默认为0（即不使用Dropout）
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(FeedForward, self).__init__()
        #self.feed_forward = nn.Sequential(...)：使用 nn.Sequential 定义一个序列模块，依次包含以下层和操作：
        #nn.Linear(d_model, d_ff)：线性变换层，将输入维度 d_model 转换为中间层维度 d_ff。
        #nn.Dropout(dropout)：Dropout层，用于随机丢弃输入张量的元素，以防止过拟合。
        #nn.GELU()：GELU激活函数层，即Gaussian Error Linear Unit，一种非线性激活函数，用于增加模型的非线性能力。
        #nn.Linear(d_ff, d_model)：再次使用线性变换层，将中间层维度 d_ff 转换回输出维度 d_model。
        #另一个 nn.Dropout(dropout)：再次应用Dropout层，用于防止模型过拟合
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(dropout),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model),
                                          nn.Dropout(dropout))
    #调用 self.feed_forward 序列模块，将输入张量 x 传入其中进行前向传播
    #返回经过前馈神经网络处理后的张量，其形状与输入张量相同 (batch_size, seq_len, d_model)
    def forward(self, x):
        return self.feed_forward(x)
    #这段代码定义了一个包含两个线性层、两个Dropout层和一个GELU激活函数的前馈神经网络模块。
    #在前向传播过程中，输入张量经过序列模块中的每一层操作，最终返回经过非线性变换的输出张量。
    #这种结构常用于Transformer模型中的每个注意力层后面的前馈神经网络部分，用于增加模型的表达能力和非线性特性。



#定义了一个名为 Image2TextGate 的PyTorch模块（Module），用于实现图像到文本的门控机制  Conv(Avg(Xˆ v ⊙ Θv))
class Image2TextGate(nn.Module):
    def __init__(self, n, d_model):
        super(Image2TextGate, self).__init__()
        self.n = n
        #定义一个平均池化层，用于对输入的第二维（C维度，通道维度）进行平均池化，池化窗口大小为 n
        self.avg_pool = nn.AvgPool1d(kernel_size=n)
        #定义一个1维卷积层，用于处理平均池化后的特征表示。该卷积层的输入和输出通道数都为 d_model，卷积核大小为1
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        #定义一个可学习的参数张量 select_para，其形状为 (n, d_model, 2)。这里使用 nn.Parameter 将其注册为模块的参数，使其能够被优化器更新
        self.select_para = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))

    def forward(self, image):
        #image：输入张量，形状为 (batch_size, n, d_model)
        B, N, C = image.shape
        #断言输入张量的第二维度大小必须等于 self.n，即门控参数的数量
        assert N == self.n
        #将输入张量 image 与复数形式的 select_para 相乘，这种操作可以看作是一种门控机制的设计，用于加权调节输入张量的信息
        image = image * torch.view_as_complex(self.select_para)
        #将张量维度进行转置，变换为 (batch_size, C, N) 的形状
        image = image.permute(0, 2, 1)  # (B, C, N)
        #对实部进行平均池化，池化结果的形状为 (batch_size, C, 1)
        image = self.avg_pool(image.real)  # (B, C, 1)
        #对平均池化后的特征表示进行1维卷积，输出形状为 (batch_size, C, 1)
        image = self.conv_layer(image)  # (B, C, 1)
        #再次对张量维度进行转置，变换为 (batch_size, 1, C) 的形状，这是最终的输出形状
        image = image.permute(0, 2, 1)  # (B, 1, C)
        #返回经过门控机制处理后的张量，形状为 (batch_size, 1, C)
        return image


#这段代码定义了一个名为 Text2ImageGate 的PyTorch模块（Module），用于实现文本到图像的门控机制 Conv(Avg(Xˆ t ⊙ Θt))
class Text2ImageGate(nn.Module):
    def __init__(self, s, d_model):
        #s：门控参数的数量；d_model：输入和输出的特征维度大小
        super(Text2ImageGate, self).__init__()
        self.s = s
        #定义一个平均池化层，用于对输入的第二维（S维度）进行平均池化，池化窗口大小为 s
        self.avg_pool = nn.AvgPool1d(kernel_size=s)
        #定义一个1维卷积层，用于处理平均池化后的特征表示。该卷积层的输入和输出通道数都为 d_model，卷积核大小为1
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        #定义一个可学习的参数张量 select_para，其形状为 (s, d_model, 2)。这里使用 nn.Parameter 将其注册为模块的参数，使其能够被优化器更新
        self.select_para = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))

    def forward(self, text):
        #text：输入张量，形状为 (batch_size, s, d_model)
        #将输入张量 text 与复数形式的 select_para 相乘，这种操作可以看作是一种门控机制的设计，用于加权调节输入张量的信息
        text = text * torch.view_as_complex(self.select_para)  # (B, S, C)
        #将张量维度进行转置，变换为 (batch_size, d_model, s) 的形状
        text = text.permute(0, 2, 1)
        #对实部进行平均池化，池化结果的形状为 (batch_size, d_model, 1)
        text = self.avg_pool(text.real)  # (B, C, 1)
        #对平均池化后的特征表示进行1维卷积，输出形状为 (batch_size, d_model, 1)
        text = self.conv_layer(text)  # (B, C, 1)
        #再次对张量维度进行转置，变换为 (batch_size, 1, d_model) 的形状，这是最终的输出形状
        text = text.permute(0, 2, 1)  # (B, 1, C)
        #返回经过门控机制处理后的张量，形状为 (batch_size, 1, d_model)
        return text
#在前向传播过程中，利用平均池化和卷积操作，结合可学习的门控参数 select_para，实现了对输入特征的加权处理，以提取和调节关键信息


# 使用之前定义的门控机制，从文本和图像数据中选择特定的频率 X˜ t = Xˆ t ⊙ Conv(Avg(Xˆ v ⊙ Θv)) (6)
# 这段代码实现了一个在频域中选择图像信息的模块 ImageFrequencySelection；
# 在前向传播过程中，首先通过 Text2ImageGate 实例处理输入的文本信息，得到门控信息 text_gate；
# 然后将输入的图像张量 image 与门控信息 text_gate 相乘，以实现对图像在频域中的选择和加权；
# 这种机制可以有效地结合文本信息对图像进行特定信息的强化或抑制。
class ImageFrequencySelection(nn.Module):
    def __init__(self, s, d_model):
        super(ImageFrequencySelection, self).__init__()
        # s：用于门控的参数数量；d_model：输入和输出的特征维度大小
        # 初始化一个 Text2ImageGate 实例作为成员变量 text_gate；这个实例用于处理输入的文本信息，以控制和调节图像在频域中的选择
        self.text_gate = Text2ImageGate(s, d_model)

    def forward(self, image, text):
        # image：输入的图像张量，在频域中表示为 (batch_size, N, C)，其中 N 是图像的高度乘以宽度，C 是通道数
        # text：输入的文本张量，形状为 (batch_size, S, d_model)，经过 Text2ImageGate 处理后得到的门控信息
        """
        image: (B, N, C)  N=h*w  in frequency domain
        """
        # 调用 Text2ImageGate 实例处理输入的文本张量 text，返回门控信息 text_gate，形状为 (batch_size, 1, d_model)
        text_gate = self.text_gate(text)
        # 将输入的图像张量 image 与门控信息 text_gate 相乘。这一步是在频域中选择和加权图像的信息
        image = image * text_gate
        # 返回经过门控机制处理后的图像张量，形状与输入的 image 相同 (batch_size, N, C)
        return image


# 使用之前定义的门控机制，从文本和图像数据中选择特定的频率  X˜ v = Xˆ v ⊙ Conv(Avg(Xˆ t ⊙ Θt))
# TextFrequencySelection 是一个继承自 nn.Module 的PyTorch模型类
class TextFrequencySelection(nn.Module):
    # __init__ 方法接受两个参数 n 和 d_model，分别表示输入的维度大小和模型的维度大小
    def __init__(self, n, d_model):
        # 在 __init__ 方法中，调用了 super().__init__() 来初始化父类 nn.Module
        super(TextFrequencySelection, self).__init__()
        # 创建了一个名为 image_gate 的成员变量，其类型为 Image2TextGate(n, d_model)。
        # 这表明 TextFrequencySelection 类包含一个名为 image_gate 的子模块，这个子模块的作用是将图像转换为文本相关的信息
        self.image_gate = Image2TextGate(n, d_model)

    def forward(self, text, image):
        # self.image_gate(image) 调用了 image_gate 模块，并传入 image 数据，得到 image_gate，
        # 这个值可以理解为通过图像得到的一种权重或者控制信号
        image_gate = self.image_gate(image)
        # 将文本数据 text 与 image_gate 相乘，这意味着通过图像门控制文本数据的一部分。
        # 这种操作通常用于根据图像内容调整文本的特定部分或者加权处理文本
        text = text * image_gate
        # 返回经过处理后的文本数据text
        return text

#定义了一个名为 AddNorm 的神经网络模块，通常用于多头自注意力机制中的加法和归一化操作
class AddNorm(nn.Module):
    #__init__ 方法接受两个参数 d_model 和 dropout（可选，默认为0.0）
    def __init__(self, d_model, dropout=0.):
        #在 __init__ 方法中，调用了 super().__init__() 来初始化父类 nn.Module
        super(AddNorm, self).__init__()
        #self.norm1 是一个 LayerNorm 层，用于对输入进行归一化，d_model 指定了归一化的维度
        self.norm1 = nn.LayerNorm(d_model)
        #self.dropout 是一个 Dropout 层，用于在训练过程中进行随机失活，以防止过拟合。dropout 参数控制失活的比例
        self.dropout = nn.Dropout(dropout)
        #self.feed_forward 是一个 FeedForward 类型的模块，用于处理输入数据。
        #这里假设 FeedForward 是一个自定义的前馈神经网络，用于对 d_model 维度的数据进行处理
        self.feed_forward = FeedForward(d_model, d_model, dropout)
        #self.norm2 是另一个 LayerNorm 层，用于在加法操作后对输出进行归一化
        self.norm2 = nn.LayerNorm(d_model)
    #forward 方法定义了数据在模型中前向传播的过程
    def forward(self, x):
        #x = self.norm1(x) 将输入数据 x 进行第一次归一化处理
        x = self.norm1(x)
        #将第一次归一化后的结果保存在 x_ 变量中，后续用于加法操作
        x_ = x
        #对归一化后的数据应用 Dropout 操作
        x = self.dropout(x)
        #将 Dropout 后的数据输入到前馈神经网络 feed_forward 中进行处理，然后加上之前保存的 x_
        #这种加法操作常见于多头自注意力机制中，用于引入残差连接
        x = self.feed_forward(x) + x_
        #x = self.norm2(x) 对加法操作后的结果再进行一次归一化处理
        x = self.norm2(x)
        #最后，返回归一化后的结果 x
        return x
#定义了一个用于多头自注意力机制的模块 AddNorm，它通过归一化、Dropout、前馈神经网络和残差连接来处理输入数据，
#以及在处理后再次归一化，最终返回处理后的结果。
#这种模块在Transformer等架构中广泛应用，用于增强模型的学习能力和稳定性


# 在文本和图像输入上执行基于傅里叶变换的操作，应用滤波器和频率选择
class FtLayer(nn.Module):
    # __init__ 方法接受多个参数：d_model 表示模型的维度大小，s 表示文本相关的维度大小，n 表示图像相关的维度大小；
    # num_filter 表示滤波器的数量，默认为2，
    # dropout 表示dropout的比例，默认为0.0，use_bank 表示是否使用滤波器，默认为True
    def __init__(self, d_model, s, n, num_filter=2, dropout=0., use_bank=True):
        super(FtLayer, self).__init__()
        self.s = s
        self.n = n
        self.use_bank = use_bank
        self.num_filter = num_filter
        # 在 __init__ 方法中，定义了多个 nn.Parameter，分别表示文本和图像的权重 text_weight 和 image_weight
        # 以及文本和图像的滤波器 text_filter_bank 和 image_filter_bank。这些参数是需要学习的模型参数

        # 定义文本（text）相关的权重和滤波器
        self.text_weight = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))
        self.text_filter_bank = nn.Parameter(torch.randn(num_filter, s, d_model, 2, dtype=torch.float32))

        # 定义图像（image）相关的权重和滤波器
        self.image_weight = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))
        self.image_filter_bank = nn.Parameter(torch.randn(num_filter, n, d_model, 2, dtype=torch.float32))

        # 创建文本频率选择和图像频率选择的模块；
        # 创建了 TextFrequencySelection 和 ImageFrequencySelection 的实例，分别用于文本和图像的频率选择
        self.text_frequency_select = TextFrequencySelection(n, d_model).to(device)
        self.image_frenquency_select = ImageFrequencySelection(s, d_model).to(device)

        # 创建文本和图像的加法归一化模块
        # 创建了 AddNorm 的实例 text_add_norm 和 image_add_norm，用于对处理后的文本和图像进行加法和归一化操作
        self.text_add_norm = AddNorm(d_model, dropout)
        self.image_add_norm = AddNorm(d_model, dropout)

    # filter 方法用于对输入数据 x 进行频率滤波操作
    def filter(self, x, length, filter_bank, weight):
        # 如果 self.use_bank 为 True，则使用滤波器 filter_bank 对输入进行滤波处理。
        # 具体过程是计算输入数据的功率，并应用余弦函数加权滤波器的处理，公式（5）
        if self.use_bank:
            power = (x * x) / length
            Y = []
            for k in range(self.num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * self.num_filter))
                Y.append(power * filter_bank[k] * cos)
            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        # 如果 self.use_bank 为 False，则直接将输入数据乘以 weight
        else:
            x = x * weight

        return x

    def forward(self, text, image, spatial_size=None):
        # 接受三个参数 text 和 image 表示输入的文本和图像数据，spatial_size 可选，表示空间大小
        # 首先保存输入数据 text 和 image 到 x_text 和 x_image 中，并进行形状断言确保输入数据的维度符合预期
        x_text = text
        B, S, D = text.shape
        assert S // 2 + 1 == self.s

        x_image = image
        B, N, C = image.shape
        assert N // 2 + 1 == self.n
        # if spatial_size:
        #     a, b = spatial_size
        # else:
        #     a = b = int(math.sqrt(N))

        # fft
        # 对 text 和 image 执行快速傅里叶变换（FFT），torch.fft.rfft 函数用于实现实数输入的快速傅里叶变换，norm='ortho' 表示使用正交归一化
        _text = torch.fft.rfft(text, dim=1, norm='ortho')
        _image = torch.fft.rfft(image, dim=1, norm='ortho')

        # 分别对 _text 和 _image 执行频率滤波操作，调用了 self.filter 方法，
        # 用 text_filter_bank 和 text_weight 对 _text 进行滤波，用 image_filter_bank 和 image_weight 对 _image 进行滤波
        # frequency filter
        _text = self.filter(_text, self.s, torch.view_as_complex(self.text_filter_bank),
                            torch.view_as_complex(self.text_weight))
        _image = self.filter(_image, self.n, torch.view_as_complex(self.image_filter_bank),
                             torch.view_as_complex(self.image_weight))
        # 调用 self.text_frequency_select 和 self.image_frenquency_select 方法，分别对 _text 和 _image 进行频率选择
        # frequency select
        _text = self.text_frequency_select(_text, _image)
        _image = self.image_frenquency_select(_image, _text)
        # 对 _text 和 _image 执行反向快速傅里叶变换（IFFT），torch.fft.irfft 函数用于实现实数输入的反向快速傅里叶变换，
        # n=S 和 n=N 分别指定输出的长度，dim=1 表示对第一个维度进行操作，norm='ortho' 表示使用正交归一化
        # ifft
        text = torch.fft.irfft(_text, n=S, dim=1, norm='ortho')
        image = torch.fft.irfft(_image, n=N, dim=1, norm='ortho')
        # image = image.view(B, N, C)
        # 最后，对处理后的文本 text 和图像 image 分别调用 self.text_add_norm 和 self.image_add_norm 执行加法和归一化操作，
        # 并返回处理后的结果 text 和 image
        # add & norm
        text = self.text_add_norm(text + x_text)
        image = self.image_add_norm(image + x_image)

        return text, image


# 这段代码定义了一个复杂的神经网络模块 FtLayer，它涉及文本和图像数据的频域处理、滤波、频率选择以及加法归一化操作。
# 这种模块通常用于处理时域数据的频域特征，


#由多个堆叠的 FtLayer 实例组成
class FtBlock(nn.Module):
    def __init__(self, d_model, s, n, num_layer=1, num_filter=2, dropout=0.):
    #__init__ 方法接受多个参数：d_model 表示模型的维度大小，s 表示文本相关的维度大小（一般是文本序列长度的一半加1），n 表示图像相关的维度大小，
    # num_layer 表示 FtLayer 的层数，默认为1，num_filter 表示滤波器的数量，默认为2，dropout 表示dropout的比例，默认为0.0
        """
        :param d_model:
        :param s: seq_len / 2 + 1
        :param h:
        :param w:
        :param n:
        """
        super(FtBlock, self).__init__()
        #在 __init__ 方法中，通过 nn.ModuleList 创建了一个名为 self.ft 的模块列表，
        #其中每个元素都是一个 FtLayer(d_model, s, n, num_filter, dropout) 的实例。
        #这样做的目的是将多个 FtLayer 组合成一个块，使得整体的频域处理可以逐层进行
        self.ft = nn.ModuleList([FtLayer(d_model, s, n, num_filter, dropout) for _ in range(num_layer)])

    def forward(self, text, image):
        for ft_layer in self.ft:
            text, image = ft_layer(text, image)

        return text, image
class ImageFrequencySelection(nn.Module):                 #公式(7)
    def __init__(self, s, d_model):
        super(ImageFrequencySelection, self).__init__()

        self.text_gate = Text2ImageGate(s, d_model)

    def forward(self, image, text):
        """
        image: (B, N, C)  N=h*w  in frequency domain
        """
        text_gate = self.text_gate(text)                  #要注意这里返回的是一个一维的向量
        image = image * text_gate                         #所以这里的*是广播式的
        return image

class TextFrequencySelection(nn.Module):                  #公式(6)
    def __init__(self, n, d_model):
        super(TextFrequencySelection, self).__init__()

        self.image_gate = Image2TextGate(n, d_model)

    def forward(self, text, image):
        image_gate = self.image_gate(image)               #同理，返回一个一维向量
        text = text * image_gate                          #广播式的element-wise乘法
        return text