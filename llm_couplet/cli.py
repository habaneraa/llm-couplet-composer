import typer
import llm_couplet
from llm_couplet.chain import CoupletComposer

app = typer.Typer()

@app.command()
def main(
    first_line: str = typer.Argument(help='上联'),
    version: bool = typer.Option(None, "--version", "-v"),
    config_path: str = typer.Argument('./model_config.yml', help='指定模型调用 API 配置路径')
):
    """大模型对联创作"""
    if version:
        print(llm_couplet.__version__)
        exit(0)
    composer = CoupletComposer.from_file(config_path)
    result = composer.compose(first_line)
    typer.echo(result)


if __name__ == "__main__":
    app()
