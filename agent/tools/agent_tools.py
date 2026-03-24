# agent的工具
import os.path
import random

from langchain_core.tools import tool
from langchain_core.utils.function_calling import tool_example_to_messages
from sympy.testing.pytest import tooslow

from rag.rag_service import RagSummaryService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummaryService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}
cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "西安", "武汉", "成都", "重庆", "苏州", "无锡", "厦门", "青岛", "济南", "郑州", "合肥", "大连", "天津", "福州", "厦门", "长沙", "南昌", "贵阳", "昆明", "武汉", "西安"]

@tool(description="从向量存储中检索与提问相关的参考资料")
def rag_summarize(query: str) -> str:
    """
    根据问题，从向量存储中检索与提问相关的参考资料
    :param query: 问题
    :return: 参考资料
    """
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """
    获取天气信息
    :param city: 城市名称
    :return: 天气信息
    """
    return f"{city}天气晴朗，温度23度"

@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    获取用户所在城市名称
    :return: 城市名称
    """
    return random.choices(cities)

@tool(description="获取用户ID，以纯字符串形式返回")
def get_user_id() -> str:
    """
    获取用户ID
    :return: 用户ID
    """
    return random.choice(user_ids)

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() ->  str:
    """
    获取当前月份
    :return: 月份
    """
    return random.choice(month_arr)

@tool(description="生成外部数据，返回字典")
def generate_external_data() -> None:
    """
    生成外部数据
    :return: 外部数据
        {
            "user_id": {
                "month" : {"特征": xxx, "效率": xxx, ...}
                "month" : {"特征": xxx, "效率": xxx, ...}
                "month" : {"特征": xxx, "效率": xxx, ...}
                ...
            },
            "user_id": {
                "month" : {"特征": xxx, "效率": xxx, ...}
                "month" : {"特征": xxx, "效率": xxx, ...}
                "month" : {"特征": xxx, "效率": xxx, ...}
                ...
            },
            "user_id": {
                "month" : {"特征": xxx, "效率": xxx, ...}
                "month" : {"特征": xxx, "效率": xxx, ...}
                "month" : {"特征": xxx, "效率": xxx, ...}
                ...
            },
            ...
        }
        :return:
        """
    # 如果外部数据为空
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        # 检测路径是否存在
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"[生成外部数据] 文件{external_data_path}不存在")


        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")
                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")
                if user_id not in external_data:
                    external_data[user_id] = {}
                # 填充字段
                external_data[user_id][time] = {
                    "特性": feature,
                    "清洁效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison
                }

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    # 生成外部数据
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[获取外部数据] 未能检索到用户{user_id}在{month}的使用记录数据")
        return ""

if __name__ == '__main__':
    # print(get_user_location())
    # generate_external_data()
    pass





