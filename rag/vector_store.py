import os
from abc import ABC

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy.codegen.ast import continue_

from utils.config_handler import chroma_conf
from model.factory import embedding_model
from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader
from utils.file_handler import txt_loader

class VectorStoreService:
    """
    向量数据库服务
    """
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            persist_directory=chroma_conf["persist_directory"],
            embedding_function=embedding_model
        )
        # 创建分词器（文本分割器）
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],  # 每个文本块的大小
            chunk_overlap=chroma_conf["chunk_overlap"],  # 文本块之间的重叠，保持上下文连贯性
            length_function=len,  # 计算文本长度的函数
            separators=chroma_conf["separators"], # 中文分割优先级
        )

    # 加载指定目录中的文档数据到向量数据库
    def load_document(self, data_dir: str = get_abs_path("data")):
        """
        加载文档到向量数据库
        :param data_dir: 数据目录路径，默认为项目根目录下的 data 文件夹
        :return: 是否成功加载
        """
        try:
            # 检查数据目录是否存在
            if not os.path.exists(data_dir):
                logger.error(f"[加载文档数据] 数据目录{data_dir}不存在")
                return False

            # 获取所有允许的文档
            allow_types = chroma_conf["allow_knowledge_file_type"]
            doc_files = listdir_with_allowed_type(data_dir, tuple(allow_types))
            # 如果没有找到允许的文档文件
            if not doc_files:
                logger.warning(f"[加载文档数据] 数据目录{data_dir}中没有找到文档文件")
                return False

            logger.info(f"[加载文档数据] 在{data_dir}中找到{len(doc_files)}个文档文件")
            # 将md5码保存至指定文件
            def save_md5_hex(file_path: str, save_path: str = get_abs_path(chroma_conf["md5_hex_store"])) -> bool:
                """
                保存文件的md5值
                :param file_path: 文件路径
                :return: None
                """
                # 确保文件路径存在
                if not os.path.exists(save_path):
                    open(save_path, "w", encoding="utf-8").close()
                if not os.path.exists(file_path):
                    logger.warning(f"[加载文档数据] 文件{file_path}不存在")
                    return False
                # 计算md5码是否存在
                md5_hex = get_file_md5_hex(file_path)
                # 遍历保存的md5码
                for line in open(save_path, "r", encoding="utf-8"):
                    # 如果md5码已经存在 返回False
                    if line.strip() == md5_hex:
                        logger.warning(f"[加载文档数据] 文件{file_path}已经存在")
                        return False
                # 保存md5码到file_path中
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(md5_hex + "\n")
                    logger.info(f"[加载文档数据] 保存文件{file_path}的md5码{md5_hex}保存成功")
                return True
            # 记录文档上传数量
            document_count = 0
            # 遍历所有文档文件
            for file_path in doc_files:
                # 如果文件已经存在
                if not save_md5_hex(file_path):
                    continue
                try:
                    # 根据文件类型选择加载器
                    if file_path.endswith('.pdf'):
                        documents: list[Document] = pdf_loader(file_path)
                    elif file_path.endswith('.txt'):
                        documents: list[Document] = txt_loader(file_path)
                    else:
                        continue

                    if not documents:
                        logger.error(f"[加载文档数据] 文档{file_path}为空，加载失败")
                        continue

                    # 使用分词器分割文档
                    split_docs: list[Document] = self.splitter.split_documents(documents)

                    # 如果分割后的文档为空
                    if not split_docs:
                        logger.error(f"[加载文档数据] 分割后的{file_path}为空，加载失败")
                        continue

                    # 添加到向量数据库
                    self.vector_store.add_documents(split_docs)

                    document_count += 1
                    logger.info(f"[加载文档数据] 成功将{file_path}添加到向量数据库")


                except Exception as e:
                    logger.error(f"[加载文档数据] 加载文档{file_path}失败：{str(e)}")
                    continue


            logger.info(f"[加载文档数据] 成功将{document_count}个文档添加到向量数据库")
            return None


        except Exception as e:
            logger.error(f"[加载文档数据] 加载文档失败：{str(e)}", exc_info=True)
            return None

    # 获取向量数据库的检索器
    def get_retriever(self):
        """
        获取向量数据库的检索器
        :return: 向量数据库的检索器
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["k"]}
        )


if __name__ == '__main__':
    vector_store = VectorStoreService()
    vector_store.load_document()
