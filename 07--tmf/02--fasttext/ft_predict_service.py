import fasttext
import jieba

from config import Config

config = Config()

# 1- 加载训练好的模型：因为词级的模型效果最好
model = fasttext.load_model(config.model_word_auto_train)


# 2- 预测函数
def predict(news_data):
    """
    对用户输入的新闻标题进行分类预测
    :param news_data: 字典。格式：{"title":新闻标题}
    :return: 字典。格式：{"title":新闻标题, "pred_class":分类预测结果名称}
    """
    # 1- 【可选】增加健壮性的代码
    if not news_data.__contains__("title"):
        news_data["error"] = "传递的参数中没有title字段"
        return news_data

    # 2- 取出新闻标题；数据预处理，也就是分词
    title = " ".join(jieba.lcut(news_data["title"]))

    # 3- 预测
    # [title]是只包含一条新闻的列表，因此FastText按“批量预测”返回结果。
    # 返回值分为标签和概率两部分，例如：
    # ([['__label__game']], [array([0.9145])])
    # 保留批量输入，还可以避开当前环境中单字符串预测触发的NumPy兼容问题。
    print(f"title --> {title}")
    pred_result = model.predict(text=[title])
    # print(type(pred_result))
    # print(pred_result)

    # 4- 取出预测结果
    # pred_result[0]：所有新闻的标签，例如 [['__label__game']]。
    # pred_result[0][0]：第一条新闻的标签列表，例如 ['__label__game']。
    # pred_result[0][0][0]：第一条新闻的第一个标签字符串，例如 '__label__game'。
    # 原来少取了一层，对列表调用replace()，所以会报AttributeError。
    result = pred_result[0][0][0].replace("__label__", "")

    # 5- 返回结果
    news_data["pred_class"] = result
    return news_data


if __name__ == "__main__":
    print("")
    news_data = {"title": "体验2D巅峰 倚天屠龙记十大创新概览"}
    # news_data = {"aaaa": "体验2D巅峰 倚天屠龙记十大创新概览"}

    result = predict(news_data)
    print(result)
