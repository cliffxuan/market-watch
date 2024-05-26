from decimal import Decimal

import boto3
import typer
import yfinance as yf

app = typer.Typer()

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TICKERS = {"SUI": "SUI20947-USD"}


# class Crypto(BaseModel):
#     symbol: str
#     history: pd.DataFrame

#     @classmethod
#     def from_item(cls, item: dict) -> "Crypto":
#         history = pd.DataFrame
#         return cls.model_validate({})


def get_table():
    return boto3.resource("dynamodb").Table("crypto")


@app.command()
def dump(symbol: str):
    symbol = symbol.upper()
    ticker = yf.ticker.Ticker(TICKERS.get(symbol, f"{symbol}-USD"))
    hist = ticker.history(period="max")[COLUMNS]  # type: ignore
    hist.index = hist.index.date.astype("str")  # type: ignore
    item = {
        row[0]: list(map(lambda f: Decimal(str(f)), row.item()[1:]))
        for row in hist.to_records()
    }

    get_table().put_item(Item={"symbol": symbol, "histroy": item})


@app.command()
def load(symbol: str):
    symbol = symbol.upper()
    get_table().get_item(Key={"symbol": symbol})


if __name__ == "__main__":
    app()
