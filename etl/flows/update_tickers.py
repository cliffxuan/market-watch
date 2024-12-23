import os
import subprocess
from pathlib import Path

from prefect import flow

PWD = Path(__file__).absolute().parent
PROJECT_DIR = PWD.parent.parent
script = PROJECT_DIR / "scripts" / "trim-git-histroy.sh"
assert script.exists()


def run_shell_command(shell_command: list[str], env_vars: dict | None = None) -> None:
    env = os.environ | (env_vars or {})

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


@flow(name="Update Tickers", log_prints=True)
def update_tickers() -> None:
    env_vars = {"GIT_SSH_COMMAND": f"ssh -i {Path.home() / '.ssh' / 'oc_cloud'}"}
    run_shell_command([str(script)], env_vars)


def main() -> None:
    update_tickers.serve(name="update_tickers daily", cron="0 8 * * *")


if __name__ == "__main__":
    main()
