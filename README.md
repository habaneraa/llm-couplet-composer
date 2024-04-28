# 📜🖌️ LLM Chinese Couplet Composer

![Static Badge](https://img.shields.io/badge/汉语-对联-red)
![Python Version](https://img.shields.io/badge/Python-3.12-orange)
![GitHub License](https://img.shields.io/github/license/habaneraa/llm-couplet-composer)
![GitHub last commit](https://img.shields.io/github/last-commit/habaneraa/llm-couplet-composer)

🏮 语言模型创作对联 - 基于 [LangChain](https://python.langchain.com/docs/get_started/introduction) 实现  🏮

## 快速开始

pip 安装

```bash
git clone git@github.com:habaneraa/llm-couplet-composer.git
cd llm-couplet-composer
pip install -e .
```

poetry 安装
```bash
git clone git@github.com:habaneraa/llm-couplet-composer.git
cd llm-couplet-composer
poetry install
```

或者，手动安装依赖，不安装此包
```bash
pip install langchain langchain-openai typer
```

配置模型调用 API：请按照 `model_config.yml.example` 手动修改，修改后删掉 `.example` 后缀

安装包后可以直接在命令行启动: `couplet <上联>` （确保你在虚拟环境内）

或者执行 Python: `python ./llm_couplet/cli.py <上联>`

## Python 批量调用 (使用协程)

```python
from llm_couplet.chain import CoupletComposer, LLMConfig

api_key = 'xxx'
base_url = 'https://api.openai.com/v1'
composer = CoupletComposer(
    LLMConfig('gpt-3.5-turbo-instruct', api_key, base_url, 0.1),
    LLMConfig('gpt-3.5-turbo', api_key, base_url, 0.7),
)

async def process_inputs(inputs: list[str]) -> list[str]:
    tasks = [asyncio.create_task(composer.acompose(input_str)) for input_str in inputs]
    return await asyncio.gather(*tasks)

上联 = ['烟锁池塘柳', '建党创军，开天辟地锤镰举', '学历非能力']
下联 = asyncio.run(process_inputs(上联))
for first, last in zip(上联, 下联):
    print(f'上联：{first}；下联：{last}')
```
