from collections import namedtuple
from operator import itemgetter

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIError

from llm_couplet.prompt_templates import (analyze_pos, analyze_senti,
                                          analyze_tones, analyze_topic,
                                          chars_prompt,
                                          compose_couplet_messages,
                                          compose_couplet_single)
from llm_couplet.utils import count_chinese_characters

LLMConfig = namedtuple("LLMConfig", ["model", "api_key", "base_url", "temperature"])


class CoupletComposer:

    def __init__(self, first_llm_config: LLMConfig, last_llm_config: LLMConfig) -> None:
        self.first_llm_config, self.last_llm_config = first_llm_config, last_llm_config
        self.chain = self.build_chain()
        self.single = self.build_single()

    def build_chain(self):
        # first_llm = OpenAI(**self.first_llm_config._asdict())
        first_llm = ChatOpenAI(**self.first_llm_config._asdict())
        last_llm = ChatOpenAI(**self.last_llm_config._asdict())
        output_parser = StrOutputParser()
        get_num_characters = lambda s: chars_prompt.format(count_chinese_characters(s))
        return (
            {
                "chars": itemgetter("first_line") | RunnableLambda(get_num_characters),
                "pos": ChatPromptTemplate.from_messages(analyze_pos) | first_llm | output_parser,
                "tones": ChatPromptTemplate.from_messages(analyze_tones)
                | first_llm
                | output_parser,
                "topic": ChatPromptTemplate.from_messages(analyze_topic)
                | first_llm
                | output_parser,
                "senti": ChatPromptTemplate.from_messages(analyze_senti)
                | first_llm
                | output_parser,
                "first_line": itemgetter("first_line"),
            }
            | ChatPromptTemplate.from_messages(compose_couplet_messages)
            | last_llm
            | output_parser
        )

    def build_single(self):
        last_llm = ChatOpenAI(**self.last_llm_config._asdict())
        output_parser = StrOutputParser()
        return ChatPromptTemplate.from_template(compose_couplet_single) | last_llm | output_parser

    def compose(self, first_line: str) -> str:
        return self.chain.invoke({"first_line": first_line})

    async def acompose(self, first_line: str) -> str:
        try:
            return await self.chain.ainvoke({"first_line": first_line})
        except APIError as e:
            return e.message

    async def acompose_single(self, first_line: str) -> str:
        try:
            return await self.single.ainvoke({"first_line": first_line})
        except APIError as e:
            return e.message

    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as f:
            llm_config = yaml.safe_load(f)
        return cls(LLMConfig(**llm_config["first_llm"]), LLMConfig(**llm_config["last_llm"]))
