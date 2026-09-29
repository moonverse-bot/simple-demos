"""
多智能体协作模块（基于 crewAI）

把「检索资料 -> 撰写回答 -> 事实核查」拆成三个 Agent 依次协作，
作为单次大模型调用的增强版：回答更严谨，来源标注更可靠。

对外主要暴露：

    run_crew(question, rag, model="deepseek/deepseek-chat", temperature=0.3) -> str
"""

from __future__ import annotations

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

DEFAULT_CREW_MODEL = "deepseek/deepseek-chat"


def to_crew_model(model: str) -> str:
    """把界面上的模型名（deepseek-chat）补成 crewAI 需要的带 provider 前缀的名字。"""
    return model if "/" in model else f"deepseek/{model}"


def _make_llm(model: str, temperature: float) -> LLM:
    """构造 LLM；deepseek-reasoner 不支持 temperature，需要跳过该参数。"""
    model = to_crew_model(model)
    if "reasoner" in model:
        return LLM(model=model)
    return LLM(model=model, temperature=temperature)


def make_knowledge_tool(rag):
    """把 RAG 检索包装成 crewAI 工具，让 Agent 能主动查资料。"""

    @tool("search_knowledge_base")
    def search_knowledge_base(query: str) -> str:
        """在用户上传的知识库中检索与 query 相关的资料片段。
        输入应为检索关键词或问题本身，返回最相关的若干片段（带编号）。
        """
        if rag.is_empty:
            return "知识库为空：用户还没有上传任何文档，请基于通用知识回答，并说明未使用资料。"
        context = rag.build_context(query, k=4)
        if not context:
            return "没有检索到与 query 相关的内容，请换关键词再试，或基于通用知识回答。"
        return context

    return search_knowledge_base


def build_crew(
    rag,
    model: str = DEFAULT_CREW_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
) -> Crew:
    """组装三 Agent 协作的 Crew：检索 -> 撰写 -> 核查。"""
    llm = _make_llm(model, temperature)
    knowledge_tool = make_knowledge_tool(rag)

    researcher = Agent(
        role="资料检索员",
        goal="从用户上传的知识库中找出与问题最相关的资料片段",
        backstory=(
            "你擅长把用户的问题拆解成关键词，并快速定位知识库中的相关段落，"
            "只汇报资料里真实存在的内容。"
        ),
        tools=[knowledge_tool],
        llm=llm,
        verbose=verbose,
    )
    writer = Agent(
        role="答案撰写员",
        goal="依据检索到的资料，用简洁清晰的中文回答用户问题",
        backstory=(
            "你是一位严谨的科普作者，只依据资料作答，不编造事实；"
            "引用资料时标注来源编号。"
        ),
        llm=llm,
        verbose=verbose,
    )
    reviewer = Agent(
        role="事实核查员",
        goal="核查回答是否有资料依据，并输出最终修订版",
        backstory="你负责挑错：删除资料中没有依据的说法，检查来源标注是否正确。",
        llm=llm,
        verbose=verbose,
    )

    research_task = Task(
        description=(
            "用户的问题是：{question}\n"
            "请调用 search_knowledge_base 工具检索相关资料，整理出与问题相关的要点。"
        ),
        expected_output=(
            "一段资料要点汇总，说明这些要点分别来自哪些片段编号；"
            "若没有命中任何资料，要明确说明这一点。"
        ),
        agent=researcher,
    )
    writing_task = Task(
        description=(
            "根据上一步检索到的资料，回答用户的问题：{question}\n"
            "若检索结果里没有可用资料，就基于通用知识回答，并说明这一点。"
        ),
        expected_output="一份简洁的中文回答；凡是依据资料的部分，在句末标注（依据资料N）。",
        agent=writer,
        context=[research_task],
    )
    review_task = Task(
        description=(
            "检查上一份回答：每个论断是否有资料支持、来源标注是否正确、有没有编造的内容。"
            "发现问题就修正，然后给出最终答案。"
        ),
        expected_output="最终修订后的中文回答，保留正确的来源标注。",
        agent=reviewer,
        context=[writing_task],
    )

    return Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=verbose,
    )


def run_crew(
    question: str,
    rag,
    model: str = DEFAULT_CREW_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
) -> str:
    """跑一次多智能体协作，返回最终答案文本。"""
    crew = build_crew(rag=rag, model=model, temperature=temperature, verbose=verbose)
    result = crew.kickoff(inputs={"question": question})
    return str(result)
