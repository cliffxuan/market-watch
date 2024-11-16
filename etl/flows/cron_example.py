import datetime as dt

from prefect import flow


@flow(name="example cron", log_prints=True)
def timestamp() -> None:
    print(dt.datetime.now())


def main() -> None:
    timestamp.serve(name="timestamp", cron="* * * * *")


if __name__ == "__main__":
    main()
