from xml.dom.minidom import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from tenacity import retry

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummaryService:
    """
    Rag服务汇总
    """
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.retriever = self.vector_store.get_retriever()
        self.model = chat_model
        self.rag_chain = self._init_chain()

    # 创建RAG链
    def _init_chain(self):
        chain = self.prompt_template | RunnableLambda(print_prompt) | self.model | StrOutputParser()
        return chain
    # 检索文档
    def retrieve_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)
    def rag_summarize(self, query: str) -> str:
        docs: list[Document] = self.retrieve_docs(query)
        # 生成参考资料
        context = ""
        count = 0
        for doc in docs:
            count += 1
            context += f'【文档参考资料{count}】:\n 内容: {doc.page_content}\n 元数据: {doc.metadata}\n'
        result = self._init_chain().invoke({"input": query, "context": context})
        return result

if __name__ == '__main__':
    rag_service = RagSummaryService()
    print(rag_service.rag_summarize("小户型适合哪些扫地机器人"))


