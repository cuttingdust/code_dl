"""
英译法示例
"""

import os
import functools
import inspect
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re

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


####################################################################################################


class MTracePoint:
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
            f"[{result}，耗时 {elapsed_ms:.3f} ms]"
        )

        # 返回False表示不吞掉异常：如果被跟踪代码出错，程序仍然按正常方式抛出异常。
        return False


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
@MTracePoint(append_message="读取、清洗并建立英法词表")
def getdata():
    print("-" * 50)
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
@MTracePoint(append_message="创建训练数据加载器")
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
        print("0-input形状-->", input.shape)
        embed = self.ebd(input)
        print("1-embed形状-->", embed.shape)

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
@MTracePoint(append_message="测试Encoder前向传播")
def use_encoder() -> None:
    # 1- 准备数据
    dataloader = get_dataloader()

    # 2- 创建编码器对象
    my_encoder = Encoder(vocab_size=english_word_n, input_size=256, hidden_size=256)
    # 将对象发送到对应的设备
    my_encoder = my_encoder.to(device)

    # 3- 遍历数据，进行前向传播
    for x, y in dataloader:
        # 3.1- 初始化隐藏状态
        hidden = my_encoder.init_hidden()

        # 3.2- 前向传播
        output, hidden = my_encoder(x, hidden)

        print(f"2-output形状-->{output.shape}")  # 1,词的个数,256
        print(f"3-hidden形状-->{hidden.shape}")  # 1,1,256

        break


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

    use_encoder()
