import jieba
import jieba.posseg as pseg

from jieba_test import content

if __name__ == "__main__":
    content = "我爱北京天安门"

    words = pseg.lcut(content)
    print(words)
