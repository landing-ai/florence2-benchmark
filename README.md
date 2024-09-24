# florence2-benchmark
Benchmark florence2 batch performance using ray[serve]

The Ray Serve engine (https://docs.ray.io/en/latest/serve/getting_started.html) is used to serve Florence2 model, and has the following features:

- Ray serve can automatically scale up / down multiple replicas based on loads
- It can utilize fraction of CPUs and GPUs based on configuration
- It can batch multiple requests based on the intuitive configuration

The benchmark code is borrowed from vllm repository:
https://github.com/vllm-project/vllm/tree/main/benchmarks

- it generates random florence2 requests using images in a folder
- the requests can be generated with an input interval parameter
- it measures latency, throughput, etc


Start the ray server with cmd:

```
serve run serve_config.yaml
```

Run benchmark against the ray server with cmd:

```
python benchmark_serving.py --images-dir PATH_OF_DATA_DIR --request-rate 7.22 --num-prompts 500
```
