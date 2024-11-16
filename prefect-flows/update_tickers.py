import subprocess
from pathlib import Path

from prefect import flow, task


def run_shell_command(shell_command: list[str]) -> None:
    process = subprocess.Popen(
        shell_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
    return_code = process.poll()
    if return_code:
        raise subprocess.CalledProcessError(return_code, shell_command)


PWD = Path(__file__).absolute().parent
script = PWD / "../scripts" / "trim-git-histroy.sh"


@task(name="succeed")
def succeed() -> None:
    run_shell_command("ls /tmp")


@task(name="succeed")
def fail() -> None:
    run_shell_command("ls /abc")


@task(name="precheck")
def precheck() -> None:
    run_shell_command([str(script), "-p"])


@task(name="remove")
def remove() -> None:
    run_shell_command([str(script), "-r"])


@task(name="fetch")
def fetch() -> None:
    run_shell_command([str(script), "-f"])


@flow(name="Update Tickers", log_prints=True)
def update_tickers() -> None:
    run_shell_command([str(script)])
    # precheck()
    # remove()


def main():
    run_shell_command("ls /tmp".split(" "))
    run_shell_command("ls /abc".split(" "))


# GIT_SSH_COMMAND='ssh -i /path/to/your/private_key' git push origin main -f
if __name__ == "__main__":
    update_tickers()
    # main()
