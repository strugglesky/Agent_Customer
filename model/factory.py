from abc import ABC, abstractmethod

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    """
    基础抽象工厂类
    """
    @abstractmethod
    def generator(self):
        pass

class ChatModelFactory(BaseModelFactory):
    """
    聊天模型工厂类
    """
    def generator(self):
        return ChatTongyi(model=rag_conf["chat_model_name"])

class EmbeddingModelFactory(BaseModelFactory):
    """
    嵌入模型工厂类
    """
    def generator(self):
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingModelFactory().generator()

if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
    print(rag_conf["embedding_model_name"])

