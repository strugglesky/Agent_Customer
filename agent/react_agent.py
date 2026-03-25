from langchain.agents import create_agent
from langchain_community.agent_toolkits.openapi.planner import create_openapi_agent
from streamlit.web.cli import main_init

from model.factory import chat_model
from agent.tools.agent_tools import rag_summarize,get_weather,get_user_location, get_user_id, get_current_month, fetch_external_data, fill_context_for_report
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            tools=[
                rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, fetch_external_data, fill_context_for_report
            ],
            system_prompt=load_system_prompts(),
            middleware=[
                monitor_tool,
                log_before_model,
                report_prompt_switch
            ],
        )

    # 执行react_agent
    def execute_stream(self, query: str):
        """
        执行流式执行
        :param query: 问题
        :return: 流式执行结果
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        for chunk in self.agent.stream(input_dict, stream_mode = 'values', context = {"is_generate_report": False}):
            latest_messages = chunk['messages'][-1]
            if latest_messages.content:
                yield latest_messages.content.strip() + '\n'

if __name__ == '__main__':
    react_agent = ReactAgent()
    res = react_agent.execute_stream("给我生成我的使用报告")
    for chunk in res:
        print(chunk, end="", flush=True )
