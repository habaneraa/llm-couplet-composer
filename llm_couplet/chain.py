from collections import namedtuple
from operator import itemgetter

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIError

from llm_couplet import prompt_templates
from llm_couplet.utils import count_chinese_characters

LLMConfig = namedtuple("LLMConfig", ["model", "api_key", "base_url", "temperature"])


def get_num_characters(s):
    return prompt_templates.chars_prompt.format(count_chinese_characters(s))


class CoupletComposer:

    def __init__(self, first_llm_config: LLMConfig, last_llm_config: LLMConfig) -> None:
        self.first_llm_config, self.last_llm_config = first_llm_config, last_llm_config
        self.chain = self.build_chain()
        self.single = self.build_single()

    def build_chain(self):
        first_llm = ChatOpenAI(**self.first_llm_config._asdict())
        last_llm = ChatOpenAI(**self.last_llm_config._asdict())
        output_parser = StrOutputParser()

        prompt_pos = ChatPromptTemplate.from_messages(prompt_templates.analyze_pos)
        prompt_tones = ChatPromptTemplate.from_messages(prompt_templates.analyze_tones)
        prompt_topic = ChatPromptTemplate.from_messages(prompt_templates.analyze_topic)
        prompt_senti = ChatPromptTemplate.from_messages(prompt_templates.analyze_senti)
        return (
            {
                "chars": itemgetter("first_line") | RunnableLambda(get_num_characters),
                "pos": prompt_pos | first_llm | output_parser,
                "tones": prompt_tones | first_llm | output_parser,
                "topic": prompt_topic | first_llm | output_parser,
                "senti": prompt_senti | first_llm | output_parser,
                "first_line": itemgetter("first_line"),
            }
            | ChatPromptTemplate.from_messages(prompt_templates.compose_couplet_messages)
            | last_llm
            | output_parser
        )

    def build_single(self):
        last_llm = ChatOpenAI(**self.last_llm_config._asdict())
        output_parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_template(prompt_templates.compose_couplet_single)
        return prompt | last_llm | output_parser

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
