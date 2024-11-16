import os
import subprocess
from pathlib import Path

from prefect import flow


def run_shell_command(shell_command: list[str], env_vars: dict | None = None) -> None:
    env = os.environ | (env_vars or {})

    print(f'run shell command {" ".join(shell_command)} with env_vars: {env_vars}')
    process = subprocess.Popen(
        shell_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in process.stdout:  # type: ignore
        print(line, end="")
    return_code = process.poll()
    if return_code:
        raise subprocess.CalledProcessError(return_code, shell_command)


@flow(name="SSH github", log_prints=True)
def update_tickers() -> None:
    env_vars = {"GIT_SSH_COMMAND": f"ssh -i {Path.home() / '.ssh' / 'oc_cloud'}"}
    run_shell_command(["ssh", "git@github.com"], env_vars)


def main() -> None:
    update_tickers()


if __name__ == "__main__":
    main()
