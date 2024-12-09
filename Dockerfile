FROM python:3.12-slim
RUN apt update -y
RUN apt install -y gcc cmake libopenblas-dev
COPY ./ /app
WORKDIR /app
RUN pip install uv
RUN uv venv
RUN uv pip install -r requirements.txt
CMD [".venv/bin/streamlit", "run", "src/market_watch/Market_Watch.py"]
