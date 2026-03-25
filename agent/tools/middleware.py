from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


# 对工具的调用进行监控
@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor] 执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor] 传入的参数: {request.tool_call['args']}")

    try:
        # 原工具函数的执行
        result = handler(request)
        logger.info(f"[tool monitor] {request.tool_call['name']} 执行成功")
        # 判断agent是否调用了fill_context_for_report工具
        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context['is_generate_report'] = True

        return result

    except Exception as e:
        logger.error(f"[tool monitor] {request.tool_call['name']} 执行失败,错误信息： {str(e)}")
        raise e

@before_model
# 模型执行前的日志记录
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None

@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    # 判断runtime中的is_generate_report是否为True
    is_report = request.runtime.context.get('is_generate_report', False)
    # 如果为True，则切换为报告生成模式
    if is_report:
        return load_report_prompts()

    return load_system_prompts()





