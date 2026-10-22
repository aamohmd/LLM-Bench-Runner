FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
COPY LICENSE ./LICENSE
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 1000 runner
COPY src/ ./src/
COPY benchmark_scoring/ ./benchmark_scoring/
COPY scripts/ ./scripts/
RUN mkdir -p data/results && chown -R runner:runner /app
USER runner
ENTRYPOINT ["python", "scripts/run_remote_benchmark.py"]
CMD ["--help"]
