FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY mercari_shopper ./mercari_shopper
COPY web ./web

RUN pip install --no-cache-dir -e ".[web]"

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "web/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
