FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
COPY LICENSE ./LICENSE
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 1000 runner
COPY src/*.py ./src/
COPY benchmark_scoring/*.py ./benchmark_scoring/
COPY scripts/run_remote_benchmark.py ./scripts/run_remote_benchmark.py
RUN mkdir -p data/results && chown -R runner:runner /app
USER runner
ENTRYPOINT ["python", "scripts/run_remote_benchmark.py"]
CMD ["--help"]
