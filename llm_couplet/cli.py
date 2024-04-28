import typer
from langchain.globals import set_debug

import llm_couplet
from llm_couplet.chain import CoupletComposer

app = typer.Typer()

def count_chinese_characters(line: str) -> int:
    num = 0
    for ch in line:
        if '\u4e00' <= ch <= '\u9fff':
            num += 1
    return num


@app.command()
def main(
    first_line: str = typer.Argument('', help="上联"),
    version: bool = typer.Option(None, "--version", "-v"),
    config_path: str = typer.Argument("./model_config.yml", help="指定模型调用 API 配置路径"),
    debug: bool = typer.Option(None, "--debug", help="输出详细执行过程"),
):
    """大模型对联创作"""
    if version:
        typer.echo('llm-couplet', llm_couplet.__version__)
        exit(0)
    if debug:
        set_debug(True)
    if count_chinese_characters(first_line) < 2:
        typer.echo('Error: 上联至少包含两个汉字')
        exit(1)
    composer = CoupletComposer.from_file(config_path)
    result = composer.compose(first_line)
    typer.echo(result)


if __name__ == "__main__":
    app()
