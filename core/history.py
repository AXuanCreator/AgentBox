#!/user/bin/env python3
# -*- coding: utf-8 -*-

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def format_messages_to_str(messages, cut: bool = False, style: bool = False):
    lines = []
    for m in messages:
        if style:
            role = '[bold blue]Human[/bold blue]' if isinstance(m, HumanMessage) else '[bold yellow]AI[/bold yellow]'
        else:
            role = 'Human' if isinstance(m, HumanMessage) else 'AI'
        content = str(m.content).replace('\n', '')
        if cut:
            content = (content[:100] + '...') if len(content) > 100 else content
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_history(messages: list, llm_summary):
    """总结历史消息"""
    msg_str = format_messages_to_str(messages)
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话历史总结为简洁的摘要，保留关键信息"),
        ("human", "{messages_str}")
    ])

    summary_chain = summary_prompt | llm_summary | StrOutputParser()
    summary_text = summary_chain.invoke({'messages_str': msg_str})

    return summary_text
