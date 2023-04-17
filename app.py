import subprocess # TYPER DOES NOT HAVE CLI RUN FUNCTION
from pathlib import Path
import typer

app = typer.Typer()

stack = """\
rich
typer
pydantic
pygwalker\
"""

other_stack = 'pip install rich typer pydantic'
pyfig_stack = 'pip install wandb '
torch_stack = 'pip install torch torchvision torchaudio'


def run_cli(cmd: str | list[str], cwd: str = '.'):
    if isinstance(cmd, str):
        cmd = cmd.split(' ')
    print('Running: ', cmd, sep='\n')
    return subprocess.run(cmd, cwd= cwd, text= True, capture_output= True)


@app.command()
def setup_python_env(python_version: str= '3.10.10', quiet: bool= False):
    out = run_cli(f'pyenv install --skip-existing{quiet * " --verbose"} {python_version}')
    print(out.check_returncode())
    out = run_cli(f'pyenv global {python_version}') # local may be better?
    # out = run_cli(f'pyenv install --plugin {" ".join(base_stack)}')
    return None


@app.command()
def setup_poetry_env(env_reset: bool= False, quiet: bool= False):
    print('Making project.toml')
    if env_reset or not Path('pyproject.toml').exists():
        run_cli(f'poetry init --no-interaction {not quiet * "--quiet"}')
        for s in stack.split('\n'):
            run_cli(f'poetry add {s}')
    return None

@app.command()
def export_env_vars():
    if not Path('.user.sh').exists():
        print('.user.sh does not exist, you might be missing env vars')
    else:
        run_cli('bash .user.sh')
        run_cli('".user.sh" >> .gitignore')
    return None

@app.command()
def setup_clients(client_dir: str = '.config/api_clients'):
    client_dir: Path = Path(client_dir)
    for client in client_dir.rglob('*-client'):
        if client.is_dir():
            run_cli(f'poetry add {client}')

@app.command()
def setup_docker_image(image_name: str= 'docker_env', file: str= 'dockerfile'):
    run_cli(f'docker build --tag {image_name} --file {file} .')
    run_cli(f'docker run -it --rm -p 5000:5000 {image_name}')

@app.command()
def setup_poetry():
    run_cli('curl -sSL https://install.python-poetry.org | python3 -')
    run_cli('echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc')
    ''' poetry install instructions
    add to path
    $HOME/.local/bin on Unix.
    '''

@app.command()
def update_zshrc():
    # export PYENV_ROOT="$HOME/.pyenv"
    # command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    # eval "$(pyenv init -)"
    # export PATH="/Users/$USER/.local/bin:$PATH"
    pass

@app.command()
def setup(python_version: str= '3.10.10', env_reset: bool= False, quiet: bool= False, install_poetry: bool= False):
    """
    usage: python app.py setup --python_version 3.10.10
    python app.py setup --install_poetry 

    step 1: python app.py setup_poetry
    

    NB: Typer does not support fstrings as inputs """
    print('Running setup from: ', Path.cwd(), ' with python version: ', python_version)
    
    if install_poetry:
        setup_poetry()
    
    setup_python_env(python_version, quiet)

    setup_poetry_env(env_reset, quiet)

    setup_clients(client_dir= '.')

    export_env_vars()

    # setup_docker_image()
    return None


if __name__ == '__main__':  
    app()

    """
    # INSTALL PYENV
    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc

    eval "$(pyenv init -)"

    
    # INSTALL POETRY 
    curl -sSL https://install.python-poetry.org | python3 -'
    echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc

    
    # STEPS
    1- install docker here https://docs.docker.com/desktop/
    2- docker build --tag docker_env --file dockerfile .
    3- docker run -it --rm -p 5000:5000 docker_env

    4- install pyenv
    5- install poetry

    6- pyenv install 3.10.10
    7- pyenv global 3.10.10
    8- poetry init --no-interaction # creates a pyproject.toml
    9- poetry add rich typer ruff rope jupyter ipykernel prefect pydantic openai httpx fastapi pygwalker
    10- poetry install
    11- poetry env use 3.10.10

    """