from collections import namedtuple
from operator import itemgetter

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI

from llm_couplet.prompt_templates import (
    analyze_chars,
    analyze_pos,
    analyze_senti,
    analyze_tones,
    analyze_topic,
    compose_couplet_messages,
)

LLMConfig = namedtuple("LLMConfig", ["model", "api_key", "base_url", "temperature"])


class CoupletComposer:

    def __init__(self, first_llm_config: LLMConfig, last_llm_config: LLMConfig) -> None:
        self.first_llm_config, self.last_llm_config = first_llm_config, last_llm_config
        self.chain = self.build_chain()

    def build_chain(self):
        first_llm = OpenAI(**self.first_llm_config._asdict())
        last_llm = ChatOpenAI(**self.last_llm_config._asdict())
        output_parser = StrOutputParser()
        return (
            {
                "chars": PromptTemplate.from_template(analyze_chars) | first_llm | output_parser,
                "pos": PromptTemplate.from_template(analyze_pos) | first_llm | output_parser,
                "tones": PromptTemplate.from_template(analyze_tones) | first_llm | output_parser,
                "topic": PromptTemplate.from_template(analyze_topic) | first_llm | output_parser,
                "senti": PromptTemplate.from_template(analyze_senti) | first_llm | output_parser,
                "first_line": itemgetter("first_line"),
            }
            | ChatPromptTemplate.from_messages(compose_couplet_messages)
            | last_llm
            | output_parser
        )

    def compose(self, first_line: str) -> str:
        return self.chain.invoke({"first_line": first_line})
    
    async def acompose(self, first_line: str) -> str:
        return await self.chain.ainvoke({"first_line": first_line})

    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as f:
            llm_config = yaml.safe_load(f)
        return cls(LLMConfig(**llm_config["first_llm"]), LLMConfig(**llm_config["last_llm"]))
