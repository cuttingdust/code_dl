"""
英译法示例
"""

import os

# 注意：1- 代码放在整个文件的最上面；2- 0需要是字符串的类型
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import re
import random
import functools
import inspect
import socket
import subprocess
import sys
import time

from tqdm import tqdm
import webbrowser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchinfo import summary
from torchview import draw_graph
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 1- 定义变量
# 运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 翻译开始标识的索引
SOS_TOKEN = 0
# 翻译结束标识的索引
EOS_TOKEN = 1
# 文件路径
file_path = os.path.join(r"data", r"eng-fra-v2.txt")
# 句子长度规范中句子的最大长度
MAX_LENGTH = 10


## 额外 辅助类型
## 日志模块 # 函数追踪模块 # torch 神经网络调试模块 #


print("")
####################################################################################################


class MPoint:
    """
    Python版本的函数执行跟踪器，作用类似C++项目中的：

        #define MPoint MTracePoint point(__FUNCTION__)

    C++版本利用局部对象的构造函数和析构函数，在进入、离开函数时自动打印日志。
    Python没有完全相同的宏和确定性析构机制，因此这里使用两种Python原生机制实现：

    1. 装饰器（推荐用于跟踪整个函数）

        @MTracePoint()
        def getdata():
            ...

    2. 上下文管理器（用于只跟踪函数内部的一段代码）

        with MTracePoint("load_data", "读取训练数据"):
            ...

    日志会自动包含：函数名、可选附加消息、执行结果和耗时。
    如果函数内部抛出异常，异常不会被吞掉，只会先记录异常类型后继续向外抛出。
    """

    def __init__(self, function_name=None, append_message=None):
        self.function_name = function_name
        self.append_message = append_message
        self.start_time = None

    def __call__(self, func):
        """让MTracePoint对象可以作为装饰器使用。"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 每次函数调用都创建一个新的跟踪对象，避免递归或多线程调用时共享开始时间。
            with type(self)(
                function_name=self.function_name or func.__qualname__,
                append_message=self.append_message,
            ):
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        """进入with代码块时执行，等价于C++ MTracePoint构造函数中的logStart。"""
        if self.function_name is None:
            # inspect.currentframe()得到当前__enter__栈帧，f_back是with所在函数的栈帧。
            caller_frame = inspect.currentframe().f_back
            self.function_name = caller_frame.f_code.co_name

        extra_message = f" {self.append_message}" if self.append_message else ""
        print(f"=== BEGIN === {self.function_name}{extra_message} Start!")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """离开with代码块时执行，等价于C++ MTracePoint析构函数中的logEnd。"""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        extra_message = f" {self.append_message}" if self.append_message else ""

        if exc_type is None:
            result = "成功"
        else:
            result = f"异常：{exc_type.__name__}: {exc_value}"

        print(
            # BEGIN有5个字符，END只有3个字符，因此在END后补3个空格，
            # 让BEGIN和END后面的===以及函数名从同一列开始。
            f"=== END   === {self.function_name}{extra_message} End! "
            # f"[{result}，耗时 {elapsed_ms:.3f} ms]"
        )

        # 返回False表示不吞掉异常：如果被跟踪代码出错，程序仍然按正常方式抛出异常。
        return False


####################################################################################################


def start_tensorboard(log_dir, port=6007):
    """
    自动启动TensorBoard，并在默认浏览器中打开计算图页面。

    平常手动启动TensorBoard使用的是：
        tensorboard --logdir ./runs --port 6007

    这个函数只是把上述操作交给Python自动完成：
    1. 检查指定端口是否已有TensorBoard服务；
    2. 没有服务时，在后台启动TensorBoard；
    3. 等待服务可以访问后，自动打开浏览器。

    :param log_dir: TensorBoard事件文件所在目录
    :param port: 本地服务端口，默认使用6007
    :return: TensorBoard计算图页面地址
    """
    host = "127.0.0.1"
    absolute_log_dir = os.path.abspath(log_dir)
    url = f"http://{host}:{port}/#graphs"

    # connect_ex()返回0，表示该端口已经有服务监听。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        is_running = client.connect_ex((host, port)) == 0

    if not is_running:
        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            absolute_log_dir,
            "--host",
            host,
            "--port",
            str(port),
        ]

        # 不把TensorBoard的运行日志混入当前示例的控制台输出。
        process_options = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if os.name == "nt":
            # Windows下隐藏后台TensorBoard进程的命令行窗口。
            process_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            # Linux和macOS下让后台服务脱离当前Python进程。
            process_options["start_new_session"] = True

        subprocess.Popen(command, **process_options)

        # 最多等待10秒，但服务一旦启动成功就立即结束等待。
        for _ in range(50):
            time.sleep(0.2)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                if client.connect_ex((host, port)) == 0:
                    is_running = True
                    break

    if is_running:
        print(f"TensorBoard已启动：{url}")
        webbrowser.open(url)
    else:
        print(
            "TensorBoard自动启动失败，可以手动执行：\n"
            f'\t"{sys.executable}" -m tensorboard.main '
            f'--logdir "{absolute_log_dir}" --port {port}'
        )

    return url


####################################################################################################


# 2- 数据清洗
def normalize_string(line):
    # 全部转小写，并且去除前后的空白字符
    line = line.lower().strip()

    # 在标点符号的前面增加空格
    line = re.sub(r"([.!?])", r" \1", line)

    # 去除特殊内容（除了26个字母和.!?），替换成空格
    line = re.sub(r"[^a-z.!?]+", r" ", line)

    return line


# 3- 数据预处理
@MPoint(append_message="读取、清洗并建立英法词表")
def getdata():
    # 1- 读取文件的所有行
    """
    大文件推荐用readline
    小文件用readline、readlines都行
    """
    with open(file_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    # print(lines)

    # 2- 循环遍历每一行，拆分得到英语句子和法语句子。得到嵌套列表，格式如下
    # [["英语句子1","法语句子1"], ["英语句子2","法语句子2"]]
    # 普通版
    sen_pairs = []
    for line in lines:
        eng_fre = line.split("\t")

        # tmp_pair的格式是：["英语句子","法语句子"]
        tmp_pair = []
        for sen in eng_fre:
            tmp_pair.append(normalize_string(sen))

        sen_pairs.append(tmp_pair)

    # 简洁版：理解
    # sen_pairs = [[normalize_string(sen) for sen in line.split("\t")] for line in lines]
    print(sen_pairs[:5])

    # 3- 分词
    # 3.1- 初始设置
    english_word2index = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN}
    english_word_n = 2
    french_word2index = {"SOS": SOS_TOKEN, "EOS": EOS_TOKEN}
    french_word_n = 2

    # 3.2- 分别对英语句子、法语句子进行分词
    for eng_fre in sen_pairs:
        # 英语
        for word in eng_fre[0].split(" "):
            # 去重处理
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1

        # 法语
        for word in eng_fre[1].split(" "):
            # 去重处理
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # print(english_word2index)

    # 3.3- 将3.2中的词典（key是单词，value是索引）改成key是索引，value是单词的形式
    english_index2word = {value: key for key, value in english_word2index.items()}
    french_index2word = {value: key for key, value in french_word2index.items()}

    return (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        sen_pairs,
    )


(
    english_word2index,
    english_index2word,
    english_word_n,
    french_word2index,
    french_index2word,
    french_word_n,
    sen_pairs,
) = getdata()


# 4- 自定义数据集Dataset
class MyPairsDataset(Dataset):
    def __init__(self, sen_pairs):
        super().__init__()
        # 设置属性值
        self.sen_pairs = sen_pairs
        self.sample_cnt = len(
            self.sen_pairs
        )  # 获得样本条数，也就是英语和法语句子对有多少对

    def __len__(self):
        # 获得样本条数
        return self.sample_cnt

    def __getitem__(self, index):
        """
        根据索引获得对应的样本数据。index是索引，从0开始
        """

        # 1- 防止index出现负数；防止index越界
        index = min(max(index, 0), self.sample_cnt - 1)

        # 2- 获得对应索引的英语句子和法语句子
        x = self.sen_pairs[index][0]  # 英语句子
        y = self.sen_pairs[index][1]  # 法语句子

        # 3- 句子分词转成词索引，最终变成张量
        """
            为什么这里只是增加了句子末尾标识EOS_TOKEN，没有增加句子开始标识SOS_TOKEN？
            答：在seq2seq+注意力机制中，不管是Encoder编码器还是Decoder解码器，都必须明确要有句子末尾标识。
               而开始标识不是必须的，同时我们在后续模型训练的时候再加上开始标识SOS_TOKEN
        """
        # 3.1- 英语句子
        x = [english_word2index[word] for word in x.split(" ")]  # 句子分词转成词索引
        x.append(EOS_TOKEN)  # 列表最后增加句子末尾标识
        x = torch.tensor(x, dtype=torch.long, device=device)  # 变成张量

        # 3.2- 法语句子
        y = [french_word2index[word] for word in y.split(" ")]  # 句子分词转成词索引
        y.append(EOS_TOKEN)  # 列表最后增加句子末尾标识
        y = torch.tensor(y, dtype=torch.long, device=device)  # 变成张量

        return x, y


# 5- 创建Dataloader
@MPoint(append_message="创建训练数据加载器")
def get_dataloader():
    # 1- 创建Dataset
    dataset = MyPairsDataset(sen_pairs)

    # 2- 创建Dataloader
    # 因为在自定义MyPairsDataset我们并没有对句子长度进行规范，因此这里的batch_size还是只能为1
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    """
        如果当前项目中，batch_size的值设置超过1，会报如下的错：
        RuntimeError: stack expects each tensor to be equal size, but got [10] at entry 0 and [6] at entry 1
    """
    # dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)

    return dataloader


####################################################################################################


# 6- 编码器：没有注意力
class Encoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.vocab_size = vocab_size  # 英语词汇表中词的个数
        self.input_size = input_size  # 词向量维度
        self.hidden_size = hidden_size  # 隐藏状态向量维度

        # 3- 搭建神经网络结构
        # 3.1- 词嵌入层
        """
            参数解释：
                num_embeddings：词汇表中词的个数（去重后的）
                embedding_dim：词向量维度
        """
        self.ebd = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.input_size
        )

        # 3.2- 循环网络层。GRU
        """
            参数解释：
                input_size：本次输入词向量维度
                hidden_size：隐藏状态向量维度
                num_layers：隐藏层层数
                batch_first：是否将batch_size放在张量的第一个位置。注意：只会调整input和output的形状，不会改变hidden的张量形状
                    例如：[seq_len,batch_size,input_size] -> [batch_size,seq_len,input_size]
        """
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, input, hidden):
        """
        前向传播。输入英语句子，让编码器理解句子的意思
        :param input: 本次输入数据，也就是单词的索引，张量形状：[batch_size,seq_len]
        :param hidden: 上一个时间步的隐藏状态，张量形状：[num_layers,batch_size,hidden_size]
        :return:
        """
        # 1- 词嵌入层：将词索引，变成词向量
        """
            输入参数input形状：[batch_size每个批次中有几个句子,seq_len每条句子中词的个数]
            结果参数embed形状：[batch_size每个批次中有几个句子,seq_len每条句子中词的个数,input_size词向量维度]
        """
        # print("0-input形状-->", input.shape)
        embed = self.ebd(input)
        # print("1-embed形状-->", embed.shape)

        # 2- GRU层
        """
            因为前面设置了batch_first为True，因此张量形状如下
                输入参数：
                    embed：[batch_size每个批次中有几个句子,seq_len每条句子中词的个数,input_size词向量维度]
                    hidden：[num_layers,batch_size,hidden_size]

                返回结果：
                    output：[batch_size每个批次中有几个句子,seq_len每条句子中词的个数,hidden_size]
                    hidden：[num_layers,batch_size,hidden_size]
        """
        output, hidden = self.gru(embed, hidden)

        return output, hidden

    def init_hidden(self):
        # 隐藏状态张量形状：[num_layer,batch_size,hidden_size]
        return torch.zeros(size=(1, 1, self.hidden_size), device=device)


# 7- 测试编码器
@MPoint(append_message="测试Encoder前向传播")
def use_encoder() -> None:
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建编码器对象
    my_encoder = Encoder(vocab_size=english_word_n, input_size=256, hidden_size=256)

    ###################################################
    print(my_encoder)

    ###################################################

    # Dataset生成的x、y以及init_hidden()返回的hidden都位于全局device上，
    # 所以正式前向传播使用的原模型也必须发送到同一个device。
    # torchview仍会在下面复制一份CPU模型进行绘图，不会移动这份原模型。
    my_encoder = my_encoder.to(device)

    # 3- 遍历数据，进行前向传播
    for x, y in dataloader:
        # 3.1- 初始化隐藏状态
        hidden = my_encoder.init_hidden()

        ########################################################################

        # 3.2- PyTorch网络结构查看工具总结
        #
        # | 对比项          | print(model)                  | torchinfo                         | torchview                        | TensorBoard                         |
        # |-----------------|-------------------------------|-----------------------------------|----------------------------------|-------------------------------------|
        # | 核心用途        | 查看模型注册了哪些网络层      | 查看每层形状、参数量和内存估算    | 查看网络连接和张量流向           | 查看交互式计算图及完整训练过程      |
        # | 输出形式        | 控制台模型文本                | 控制台统计表格                    | PNG/SVG等静态结构图              | 浏览器中的交互式网页                |
        # | 输入输出形状    | 通常不显示                    | 详细显示                          | 在图中显示                       | 可在计算图节点中查看                |
        # | 参数量统计      | 不统计总量                    | 详细统计                          | 不是主要功能                     | 可通过其他面板间接记录              |
        # | 网络连接关系    | 只显示模块层级                | 只显示表格层级                    | 清晰、直观                       | 可缩放、展开和查看节点              |
        # | 训练指标记录    | 不支持                        | 不支持                            | 不支持                           | 支持loss、accuracy、lr等            |
        # | 参数/梯度分布   | 不支持                        | 不支持                            | 不支持                           | 支持Histograms等面板                |
        # | 是否执行forward | 否                            | 是（传入input_data时）             | 是                               | 是（调用add_graph时）               |
        # | 是否需要输入    | 否                            | 是（需要实际分析形状时）           | 是                               | 是（记录计算图时）                  |
        # | 是否需要浏览器  | 否                            | 否                                | 否                               | 是                                  |
        # | 额外依赖        | 无，PyTorch自带               | torchinfo                         | torchview、graphviz、Graphviz程序 | tensorboard                         |
        # | 主要优点        | 最快、最简单、没有执行副作用  | 形状和参数统计清晰、适合排错      | 数据流直观、图片适合文档         | 功能完整、可长期记录和比较实验      |
        # | 主要缺点        | 看不到真实数据流和实际形状    | 复杂分支不如图形直观              | 大模型图片可能过大、依赖较多     | 较重，需要事件文件和本地Web服务     |
        # | 最适合的场景    | 快速确认模型层是否定义正确    | 排查维度错误、评估模型参数规模    | 学习或检查复杂网络连接           | 正式训练监控、实验对比和计算图查看  |
        #
        # 注意：
        #   1. torchinfo、torchview和TensorBoard.add_graph()都会执行forward()。
        #      如果forward()中写了print()、计数或修改状态等操作，这些操作也会发生。
        #   2. 本示例为避免结构分析影响正式模型，分别使用deepcopy()复制模型，
        #      再使用detach().cpu()准备只用于检查的输入张量。
        #   3. TensorBoard不只是“画网络图”的工具。它更重要的用途是在训练循环中，
        #      持续记录不同epoch的loss、accuracy、learning rate、参数和梯度分布。
        #
        # 推荐选择：
        #
        # | 当前需求                                 | 推荐工具       |
        # |------------------------------------------|----------------|
        # | 只想快速知道模型由哪些层组成             | print(model)   |
        # | 想检查每层输入输出形状以及参数量         | torchinfo      |
        # | 想获得一张直观、可保存的网络连接图       | torchview      |
        # | 想在网页中查看图并持续观察训练指标       | TensorBoard    |
        #
        # 实际项目推荐顺序：
        #   print(model) -> torchinfo -> 必要时使用torchview -> 训练时使用TensorBoard

        ########################################################################

        # # 3.2- 使用当前批次的真实x和hidden生成Encoder网络结构图
        # #
        # # torchview会实际调用一次模型的forward()。为了保证绘图过程不会改变
        # # 原来的my_encoder、x和hidden，这里分别创建只用于绘图的副本。
        # #
        # # deepcopy()会复制出一个独立的Encoder对象：
        # #   my_encoder：继续用于下面真正的前向传播
        # #   graph_encoder：只用于torchview生成结构图
        # graph_encoder = copy.deepcopy(my_encoder).cpu()
        # graph_encoder.eval()
        #
        # # x和hidden的形状直接来自当前DataLoader批次，不再硬编码句子长度、
        # # batch_size或hidden_size。detach()表示绘图不需要记录原张量的梯度关系，
        # # cpu()则保证绘图不依赖CUDA和cuDNN。
        # graph_x = x.detach().cpu()
        # graph_hidden = hidden.detach().cpu()
        #
        # encoder_graph = draw_graph(
        #     model=graph_encoder,
        #     input_data=(graph_x, graph_hidden),
        #     graph_name="Seq2SeqEncoder",
        #     device="cpu",
        #     show_shapes=True,
        #     expand_nested=True,
        #     roll=True,
        #     graph_dir="TB",
        #     save_graph=False,
        # )
        #
        # # draw_graph()返回Graphviz图对象，再将它渲染成PNG文件。
        # # cleanup=True表示生成PNG后删除中间的Graphviz源文件。
        # encoder_graph.visual_graph.render(
        #     filename="seq2seq_encoder_structure",
        #     directory=".",
        #     format="png",
        #     cleanup=True,
        # )
        #
        # print("Encoder网络结构图已生成：seq2seq_encoder_structure.png")

        ########################################################################

        # # 3.2- 使用torchinfo打印Encoder的分层结构和参数统计
        # #
        # # torchinfo和上面的torchview用途不同：
        # #   torchview：把网络的计算过程画成图片；
        # #   torchinfo：在控制台打印每一层的输入形状、输出形状、参数量等信息。
        # #
        # # summary()会真正执行一次模型的forward()。为了让这次结构检查不影响
        # # 下面正式使用的my_encoder，这里重新复制一份只供torchinfo使用的模型。
        # # 这一段是独立示例，不依赖上面已经注释掉的torchview代码。
        # summary_encoder = copy.deepcopy(my_encoder).cpu()
        # summary_encoder.eval()
        #
        # # 不使用size=(1, 7)或size=(1, 1, 256)等硬编码形状。
        # # x和hidden直接取自当前DataLoader批次，因此句子长度、批次大小和
        # # 隐藏层维度发生变化时，torchinfo收到的输入也会自动跟着变化。
        # #
        # # detach()：torchinfo只负责查看网络结构，不需要连接原来的计算图；
        # # cpu()：summary_encoder已经放在CPU上，所以输入也必须放在CPU上。
        # summary_x = x.detach().cpu()
        # summary_hidden = hidden.detach().cpu()
        #
        # print("\n-------------------- torchinfo网络结构摘要 --------------------")
        # summary(
        #     model=summary_encoder,
        #     # Encoder.forward(input, hidden)需要两个位置参数，所以这里传入二元组。
        #     input_data=(summary_x, summary_hidden),
        #     # 显示到第2层，能够看到Encoder下面的Embedding层和GRU层。
        #     depth=2,
        #     # 指定表格中需要展示的列。
        #     col_names=("input_size", "output_size", "num_params", "trainable"),
        #     # 使用模型中定义的变量名显示层名称，例如ebd和gru。
        #     row_settings=("var_names",),
        #     # verbose=1表示直接把结构摘要打印到控制台。
        #     verbose=1,
        # )
        # print("---------------------------------------------------------------\n")

        ########################################################################

        # # 3.2- 使用TensorBoard记录并显示Encoder计算图
        # #
        # # 三个网络查看工具的侧重点不同：
        # #   torchinfo：在控制台查看每层的形状和参数量；
        # #   torchview：生成一张静态的网络结构图片；
        # #   TensorBoard：在浏览器中交互式查看计算图。
        # tensorboard_log_dir = os.path.join("runs", "seq2seq_encoder")
        #
        # # add_graph()会真实调用一次forward()。复制一份CPU模型专门生成计算图，
        # # 避免移动或改变下面正式前向传播使用的my_encoder。
        # tensorboard_encoder = copy.deepcopy(my_encoder).cpu()
        # tensorboard_encoder.eval()
        #
        # # 直接使用当前DataLoader批次产生的真实张量，不硬编码任何形状。
        # tensorboard_x = x.detach().cpu()
        # tensorboard_hidden = hidden.detach().cpu()
        #
        # # SummaryWriter把计算图写入TensorBoard事件文件。
        # # with代码块结束时会自动刷新并关闭文件。
        # with SummaryWriter(log_dir=tensorboard_log_dir) as writer:
        #     # Encoder.forward(input, hidden)有两个输入，因此这里传入二元组。
        #     writer.add_graph(
        #         model=tensorboard_encoder,
        #         input_to_model=(tensorboard_x, tensorboard_hidden),
        #     )
        #
        # print(f"TensorBoard计算图已写入：{os.path.abspath(tensorboard_log_dir)}")
        #
        # # 自动启动后台服务并打开Graphs页面；端口已有服务时不会重复启动。
        # start_tensorboard(log_dir=tensorboard_log_dir, port=6007)

        ########################################################################
        # 3.3- 前向传播
        output, hidden = my_encoder(x, hidden)

        print(f"2-output形状-->{output.shape}")  # 1,词的个数,256
        print(f"3-hidden形状-->{hidden.shape}")  # 1,1,256

        break


# 8- 解码器：无注意力机制
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        :param vocab_size:  解码器中词汇表大小，也就是法语词的个数4345个
        :param hidden_size: 隐藏层隐藏状态的向量维度。
            因为是人为设置embedding_dim和hidden_size，因此该参数也表示词向量的维度。但是你可以设置的不一致
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 3- 搭建网络结构
        # 3.1- 词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )

        # 3.2- 循环网络层：GRU
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # 3.3- 线性层
        self.out = nn.Linear(in_features=self.hidden_size, out_features=vocab_size)

        # 3.4- 输出的激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """
        前向传播。翻译得到法语词
        :param input: 上一个时间步的预测法语词的词索引（也就是要将上个时间步预测结果作为本次输入使用），
            张量形状：[1每个批次中法语句子的条数,1每次只传递一个法语词]。数据内容举例：[[法语词索引]]
        :param hidden: 上一个时间步的隐藏状态，张量形状：[num_layers,batch_size,hidden_size]
        :return:
        """
        print(f"1-input形状-->{input.shape}")
        print(f"2-hidden形状-->{hidden.shape}")
        # 1- 调用词嵌入层：输入词索引，得到词向量
        ebd = torch.relu(self.embedding(input))
        print(f"3-ebd形状-->{ebd.shape}")

        # 2- 调用GRU
        """
            因为前面__init__中设置了batch_first为True，因此只会影响ebd和output的张量形状

            输入参数形状：
                ebd：    [batch_size,seq_len,hidden_size]
                hidden： [num_layers,batch_size,hidden_size]

            返回结果形状：
                output： [batch_size,seq_len,hidden_size]
                hidden： [num_layers,batch_size,hidden_size]
        """
        output, hidden = self.gru(ebd, hidden)
        print(f"4-output形状-->{output.shape}")

        # 3- 调用线性层和激活函数，得到预测的概率值
        # 3.1- 将output降维为二维：因为线性层只能处理二维数据
        """
            output[-1]针对：分类场景，也就是N vs 1的情况。取得是最后一个时间步的隐藏状态。一条句子彻底理解完以后才得到最终的预测结果
            output[0]针对：翻译、文本生成的场景，也就是N vs M的情况。取得是第一个时间步的隐藏状态。一次只生成一个词，并将该词作为下一步的输入
        """
        output = output[0]
        print(f"5-output[0]形状-->{output.shape}")

        # 3.2- 调用线性层
        output = self.softmax(self.out(output))

        # 4- 返回
        return output, hidden

    def init_hidden(self):
        return torch.zeros(size=(1, 1, self.hidden_size), device=device)


# 9- 测试解码器
@MPoint(append_message="测试无注意力机制的解码器")
def use_decoder() -> None:
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建编码器和解码器
    my_encoder = Encoder(vocab_size=english_word_n, input_size=256, hidden_size=256).to(
        device=device
    )
    print("my_encoder --> ")
    print(my_encoder)

    my_decoder = Decoder(vocab_size=french_word_n, hidden_size=256).to(device=device)
    print("my_decoder --> ")
    print(my_decoder)

    # 3- 遍历数据加载器进行翻译
    for x, y in dataloader:
        # x是英语句子，里面存放的是单词的索引。张量形状[batch_size,seq_len]
        # y是法语句子，里面存放的是单词的索引。张量形状[batch_size,seq_len]

        # 3.1- 先调用编码器
        h0 = my_encoder.init_hidden()
        print(f"h0-->{h0.shape}")
        output, h0 = my_encoder(x, h0)

        print(f"y-->{y.shape}")
        # 3.2- 再调用解码器。要一个一个法语词传递进解码器
        for t in range(y.shape[1]):
            # 3.3- 先提取法语词的索引，再将索引变成二维张量，最终形状是[[法语词索引]]
            french_word_index_2dtensor = y[0][t].reshape(1, -1)
            print(f"y={y}，y[0]={y[0]}，y[0][t]={y[0][t]}")
            print(f"french_word_index_2dtensor={french_word_index_2dtensor}")

            # 3.4- 调用解码器
            output, hidden = my_decoder(french_word_index_2dtensor, h0)

            print(f"output形状-->{output.shape}")

        break  # 只跑一对英语句子和法语句子


# 10- 解码器：有注意力机制
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_p=0.1):
        """
        :param vocab_size:  解码器中词汇表大小，也就是法语词的个数4345个
        :param hidden_size: 隐藏层隐藏状态的向量维度。
            因为是人为设置embedding_dim和hidden_size，因此该参数也表示词向量的维度。但是你可以设置的不一致
        :param dropout_p 失活性 简化结构 提高效率
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.drouput_p = dropout_p

        # 3- 搭建网络结构
        # 3.1- 词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        # 词嵌入层后面的随机失活层
        self.dropout = nn.Dropout(p=self.drouput_p)

        # 3.2- 线性层：用来计算Q和K拼接后的相似性
        self.attn = nn.Linear(
            in_features=self.hidden_size + self.hidden_size, out_features=MAX_LENGTH
        )

        # 3.3- 线性层：拼接【上一个时间步翻译结果法语词的词向量】和【专属信息包】，然后将张量形状调整为GRU要求的形状
        self.attn_combine = nn.Linear(
            in_features=self.hidden_size + self.hidden_size, out_features=hidden_size
        )

        # 3.4- 循环网络层：GRU
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # 3.5- 线性层：输出层
        self.out = nn.Linear(in_features=self.hidden_size, out_features=vocab_size)

        # 3.6- 输出的激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, key, value):
        """
        :param input: 上一个时间步的预测法语词的词索引。张量形状：[batch_size,seq_len]。seq_len后续我们是一个个法语词进行传递
        :param key: 上一个时间步的隐藏状态，也就是流程图中的prev_hidden。张量形状：[num_layers,batch_size,hidden_size]
        :param value: 编码器端多个词的隐藏状态拼接后的张量。张量形状：[batch_size,seq_len是固定值为MAX_LENGTH,hidden_size]
        :return:
        """
        # 1- input转成词向量，也就是得到Q
        # 方式一：分步骤的版本
        # ebd = self.embedding(input)
        # embedded = self.dropout(ebd)

        # 方式二：合并的版本
        Q = self.dropout(self.embedding(input))

        # 2- Q和K拼接；算相似性；相似性转成权重
        # 方式一：分步骤的版本
        qk_concat = torch.cat((Q, key), dim=-1)  # Q和K拼接
        qk_score = self.attn(qk_concat)  # 算相似性
        attn_weights = torch.softmax(qk_score, dim=-1)  # 相似性转成权重

        # 方式二：合并的版本
        # attn_weights = torch.softmax(self.attn(torch.concat((Q,key),dim=-1)), dim=-1)

        # 3- 权重矩阵和value进行矩阵乘法运算，得到专属信息包
        attn_applied = torch.bmm(attn_weights, value)

        # 4- 【上一个时间步的预测法语词的词向量】和【专属信息包】拼接；调整拼接后的形状，达到GRU的输入要求
        qc_contact = torch.cat(
            (Q, attn_applied), dim=-1
        )  # 【上一个时间步的预测法语词的词向量】和【专属信息包】拼接

        gru_input = torch.relu(self.attn_combine(qc_contact))

        # 5- 调用GRU
        output, hidden = self.gru(gru_input, key)

        # 6- 计算预测概率
        output = output[0]
        output = self.softmax(self.out(output))

        # 7- 返回结果
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(size=(1, 1, self.hidden_size), device=device)


# 11- 测试解码器
@MPoint(append_message="测试有注意力机制的解码器")
def use_attn_decoder() -> None:
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建编码器和解码器
    my_encoder = Encoder(vocab_size=english_word_n, input_size=256, hidden_size=256).to(
        device=device
    )
    print("my_encoder --> ")
    print(my_encoder)

    my_decoder = AttnDecoder(vocab_size=french_word_n, hidden_size=256).to(
        device=device
    )
    print("my_decoder --> ")
    print(my_decoder)

    # 3- 遍历数据加载器进行翻译
    for x, y in dataloader:
        # x是英语句子，里面存放的是单词的索引。张量形状[batch_size,seq_len]
        # y是法语句子，里面存放的是单词的索引。张量形状[batch_size,seq_len]

        # 3.1- 先调用编码器
        # 3.1.1- 解码器
        h0 = my_encoder.init_hidden()
        print(f"h0-->{h0.shape}")
        output, hidden = my_encoder(x, h0)

        # 3.1.2- 句子长度规范处理。统一到词个数为10的长度
        # 初始化一个[1,MAX_LENGTH,256]的全0张量
        value = torch.zeros(size=(1, MAX_LENGTH, 256), device=device)
        # 计算需要复制的词个数
        copy_len = min(MAX_LENGTH, x.shape[1])  # x.shape[1]获取当前英语句子中词的个数

        # 将已有的数据复制到value中
        """
            为什么用output而不使用hidden来取隐藏状态信息数据？
            答：output记录的是最后一层，每个时间步的隐藏状态
               hidden记录的是每一层，最后一个时间步的隐藏状态
               但是value，是编码器端每个词的隐藏状态的拼接
        """
        value[:, :copy_len, :] = output[:, :copy_len, :]

        print(f"y-->{y.shape}")
        # 3.2- 再调用解码器。要一个一个法语词传递进解码器
        for t in range(y.shape[1]):
            # 3.3- 先提取法语词的索引，再将索引变成二维张量，最终形状是[[法语词索引]]
            french_word_index_2dtensor = y[0][t].reshape(1, -1)
            print(f"y={y}，y[0]={y[0]}，y[0][t]={y[0][t]}")
            print(f"french_word_index_2dtensor={french_word_index_2dtensor}")

            # 3.4- 调用解码器
            output, hidden, attn_weights = my_decoder(
                french_word_index_2dtensor, hidden, value
            )

            print(f"解码output.shape: {output.shape}")  # [1, 4345]
            print(f"解码hidden.shape: {hidden.shape}")  # [1, 1, 256]
            print(f"解码att_weights: {attn_weights}")
            print(f"解码att_weights和: {attn_weights.sum()}")
            print(f"解码att_weights.shape: {attn_weights.shape}")  # [1, 1, 10]

            print("-" * 30)
        break  # 只跑一对英语句子和法语句子


# 12- 模型训练
# 12.1- 单次的训练过程
def train_iters(x, y, my_encoder, my_decoder, encoder_adam, decoder_adam, loss) -> None:
    """
    单对样本的训练代码
    :param x: 英语句子，特征数据，形状：[batch_size,seq_len]
    :param y: 法语句子，目标值，形状：[batch_size,seq_len]
    :param my_encoder: 编码器
    :param my_decoder: 解码器
    :param encoder_adam: 编码器优化器
    :param decoder_adam: 解码器优化
    :param loss:损失函数对象
    :return: 单条样本数据的平均损失值
    """
    # 1- 调用编码器
    encoder_hidden = my_encoder.init_hidden()
    encoder_output, encoder_hidden = my_encoder(x, encoder_hidden)

    # 2- 句子长度规范
    # 2.1- 初始化全零的张量，形状[batch_size,MAX_LENGTH,hidden_size]
    value = torch.zeros(size=(1, MAX_LENGTH, 256), device=device)
    # 2.2- 计算复制的长度
    copy_len = min(MAX_LENGTH, x.shape[1])
    # 2.3- 复制值
    value[:, :copy_len, :] = encoder_output[:, :copy_len, :]

    # 3- 参数设置
    # 3.1- 将 编码器最后一个时间步的隐藏状态作为解码器的初始隐藏状态使用
    decoder_hidden = encoder_hidden
    # 3.2- 在法语句子的前面增加SOS_TOKEN，标记翻译工作的开始
    input_y = torch.tensor([[SOS_TOKEN]], device=device)
    # 3.3- 初始损失值
    loss_value = torch.tensor(data=0.0, device=device)
    # 3.4- 获得当前法语句子中法语词的个数
    y_len = y.shape[1]
    # 3.5- teacher_forcing机制的使用标识
    teacher_forcing_flag = True if random.random() < 0.5 else False

    # 4- 调用解码器
    if teacher_forcing_flag:
        # 4.1- 使用teacher_forcing机制

        for t in range(y_len):
            # 解码器的前向传播
            # 注意：输入进去和得到的隐藏状态的变量名称要完全相同，为了持续的将隐藏状态往下个时间步传递
            output, decoder_hidden, attn_weights = my_decoder(
                input_y, decoder_hidden, value
            )

            # 计算损失值
            y_true = y[0][t].reshape(1)  # 法语真实值的词索引
            loss_value += loss(output, y_true)

            # 告知解码器下一个时间步真实的目标值是多少
            input_y = y[0][t].reshape(1, -1)  # [[词索引]]
    else:
        # 4.2- 不使用。普通的训练过程
        for t in range(y_len):
            # 解码器的前向传播
            # 注意：输入进去和得到的隐藏状态的变量名称要完全相同，为了持续的将隐藏状态往下个时间步传递
            output, decoder_hidden, attn_weights = my_decoder(
                input_y, decoder_hidden, value
            )

            # 计算损失值
            y_true = y[0][t].reshape(1)  # 法语真实值的词索引
            loss_value += loss(output, y_true)

            # 将上一个时间步的预测结果（topi.detach()），作为【本次输出input_y】传递给到下一个时间步
            topv, topi = output.topk(1)  # 只取预测概率最高的那个词
            y_pred_index = topi.detach()
            if y_pred_index == EOS_TOKEN:
                # 说明到了句子的结尾。那么就结束翻译工作怪
                break
            input_y = y_pred_index

    # 5- 反向传播固定代码
    # 梯度清零
    encoder_adam.zero_grad()
    decoder_adam.zero_grad()

    # 反向传播
    loss_value.sum().backward()

    # 更新参数
    encoder_adam.step()
    decoder_adam.step()

    # 6- 返回平均损失
    return loss_value.item() / y_len


# 12.2- 整体训练流程
def train() -> None:
    # 1. 获取数据集加载器对象
    dataloader = get_dataloader()

    # 2. 模型初始化
    size = 256
    my_encoder = Encoder(
        vocab_size=english_word_n, input_size=size, hidden_size=size
    ).to(device=device)
    my_decoder = AttnDecoder(vocab_size=french_word_n, hidden_size=size).to(
        device=device
    )

    # 3. 优化器初始化
    encoder_adam = torch.optim.Adam(params=my_encoder.parameters(), lr=1e-4)
    decoder_adam = torch.optim.Adam(params=my_decoder.parameters(), lr=1e-4)

    # 4. 损失函数初始化. 使用 NLLLoss() 负数对数似然损失
    loss = nn.NLLLoss()

    # 5. 训练参数初始化
    epochs = 1
    plot_avg_loss_list = []  # 平均损失值列表，用来画图

    # 6. 循环训练
    for epoch in range(epochs):
        # 6.1 外层循环, 控制训练轮数

        plot_total_loss = 0.0
        print_total_loss = 0.0

        # 6.2 内层循环, 遍历数据集的每个样本(即: 每轮具体的训练过程)
        for index, (x, y) in enumerate(tqdm(dataloader), start=1):
            """
            index：样本条数
            x：英语句子
            y：法语句子
            """
            # 具体训练过程看下面的步骤
            myloss = train_iters(
                x,
                y,
                my_encoder,
                my_decoder,
                encoder_adam=encoder_adam,
                decoder_adam=decoder_adam,
                loss=loss,
            )

            plot_total_loss += myloss
            print_total_loss += myloss

            # 6.3 记录损失用于绘图(每100个样本记录一次)
            if index % 100 == 0:
                # 计算平均损失 = 总损失 / 100
                avg_loss = plot_total_loss / 100
                # 记录指标
                plot_avg_loss_list.append(avg_loss)
                plot_total_loss = 0.0

            # 6.4 打印训练日志 (每1000个样本打印一次)
            if index % 1000 == 0:
                # 计算平均损失 = 总损失 / 1000
                avg_loss = print_total_loss / 1000
                print(f"第{epoch+1}轮次，已训练的样本条数{index}，平均损失是{avg_loss}")
                print_total_loss = 0.0

            # if index > 1000:
            #     break

        torch.save(my_encoder.state_dict(), r"model/encoder.pkl")
        torch.save(my_decoder.state_dict(), r"model/decoder.pkl")

        # 8. 训练结束后, 绘制损失曲线
        plt.figure()
        plt.plot(plot_avg_loss_list)
        plt.show()


# 13- 模型预测
# 13.1- 单条英语句子的翻译
def seq2seq_evaluate(x, my_encoder, my_decoder):
    with torch.no_grad():
        # 1- 调用编码器：让模型理解英语句子的意思
        encoder_hidden = my_encoder.init_hidden()
        encoder_output, encoder_hidden = my_encoder(x, encoder_hidden)

        # 2- 句子长度规范
        # 2.1- 初始化全零的张量。形状：[1,MAX_LENGTH,256]
        value = torch.zeros(size=(1, MAX_LENGTH, 256), device=device)
        # 2.2- 计算拷贝的长度
        copy_len = min(MAX_LENGTH, x.shape[1])
        # 2.3- 复制数据
        value[:, :copy_len, :] = encoder_output[:, :copy_len, :]
        # 3- 定义变量
        # 3.1- 解码器的初始隐藏状态：来自编码器端最后一个时间步的隐藏状态
        decoder_hidden = encoder_hidden
        # 3.2- 翻译工作的开始
        input_y = torch.tensor([[SOS_TOKEN]], device=device)
        # 3.3- 记录解码器端每个时间步对应的权重分布
        decoder_attention = torch.zeros(
            size=(MAX_LENGTH, MAX_LENGTH), device=device
        )  # 形状[10个法语词,10个权重值]
        # 3.4- 翻译的法语词列表
        french_word_list = []

        # 4- 开始翻译
        for t in range(MAX_LENGTH):
            # 4.1- 调用解码器
            output, decoder_hidden, attn_weights = my_decoder(
                input_y, decoder_hidden, value
            )
            # 4.2- 记录权重的分布
            decoder_attention[t] = attn_weights
            # 4.3- 得到预测概率最高的那一个法语词
            # 4.3.1- 先得到最高词的词索引
            topv, topi = output.topk(1)
            french_word_index = topi.squeeze().item()
            print(f"topi-->{topi}-->{topi.squeeze()}-->{topi.squeeze().item()}")

            # 4.3.2- 判断是否到了句子的结束标识
            if french_word_index == EOS_TOKEN:
                french_word_list.append("EOS")
                break
            else:
                # 4.3.3- 词索引转成文字内容
                french_word_list.append(french_index2word[french_word_index])

            # 4.3.4- 本次的预测结果法语词索引作为下个时间步的输入数据使用
            input_y = torch.tensor([[french_word_index]], device=device)

        return french_word_list, decoder_attention


def use_seq2seq_evaluate():
    # 1 - 准备数据
    # 因为以后是手动准备需要翻译的数据，因此不需要加载数据
    # get_dataloader()

    # 2- 加载训练好的模型
    size = 256
    my_encoder = Encoder(
        vocab_size=english_word_n, input_size=size, hidden_size=size
    ).to(device=device)
    my_encoder.load_state_dict(torch.load(r"model/encoder.pkl"))

    my_decoder = AttnDecoder(vocab_size=french_word_n, hidden_size=size).to(
        device=device
    )
    # weights_only：只从训练好的模型中加载权重信息，其他信息不加载。作用：加快模型的加载速度
    my_decoder.load_state_dict(torch.load(r"model/decoder.pkl"))

    # 3- 准备数据
    my_samplepairs = [
        [
            "i m impressed with your french .",
            "je suis impressionne par votre francais .",
        ],
        ["i m more than a friend .", "je suis plus qu une amie ."],
        ["she is beautiful like her mother .", "elle est belle comme sa mere ."],
    ]

    # 4- 翻译
    for index, pair in enumerate(my_samplepairs):
        x = pair[0]  # 英语句子
        y = pair[1]  # 法语句子

        # 4.1- 分词并且转成向量
        x = [english_word2index[word] for word in x.split(" ")]
        x.append(EOS_TOKEN)
        x_tensor = torch.tensor(x, dtype=torch.long, device=device).reshape(1, -1)

        # 4.2- 调用预测代码
        french_word_list, decoder_attention = seq2seq_evaluate(
            x_tensor, my_encoder, my_decoder
        )

        # 4.3- 打印结果内容
        print(f"真实法语句子：{y}")
        print(f"预测法语句子：{' '.join(french_word_list)}")
        print(f"权重矩阵：{decoder_attention.shape}")
        print(f"权重矩阵：{decoder_attention.sum()}")
        print(f"权重矩阵：{decoder_attention}")

        # 4.4- 绘制权重矩阵
        # Matplotlib底层使用NumPy，只能处理CPU内存中的数据。
        # decoder_attention当前位于cuda:0，因此绘图前需要：
        # 1. detach()：与PyTorch计算图分离；
        # 2. cpu()：从GPU显存复制到CPU内存；
        # 3. numpy()：转换成NumPy数组。
        attention_numpy = decoder_attention.detach().cpu().numpy()
        plt.matshow(attention_numpy)
        plt.show()

        print("-" * 30)


if __name__ == "__main__":

    # content = " i LOVE heima! "
    # content = " i LOVE hei@ma! "
    # print(f"-{normalize_string(content)}-")

    # getdata()

    # print("-" * 50)
    # dataloader = get_dataloader()
    # for x, y in dataloader:
    #     print(f"英语句子-->{x.shape}-->{x}")
    #     print(f"法语句子-->{y.shape}-->{y}")
    #
    #     break
    #
    # print("-" * 50)
    # print(english_word_n)
    # print("-" * 50)
    # print(english_word2index)
    # print("-" * 50)
    # print(english_index2word)
    #
    # print("-" * 50)
    # print(french_word_n)
    # print("-" * 50)
    # print(french_word2index)
    # print("-" * 50)
    # print(french_index2word)

    # use_encoder()

    print(french_word_n)

    # use_decoder()

    # use_attn_decoder()

    # train()

    use_seq2seq_evaluate()
