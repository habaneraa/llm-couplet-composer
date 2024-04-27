# 📜🖌️ LLM Chinese Couplet Composer

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
