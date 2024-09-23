"""
Benchmark online non-streaming serving throughput and latency.

On the server side, run ray serve command:
    serve run serve_config.yaml


On the client side, run:
    python benchmark_serving.py
        --images-dir <path to image>
        --request-rate <request_rate> # By default <request_rate> is 10 / sec
        --num-prompts <num_prompts> # By default <num_prompts> is 500
"""

import argparse
import asyncio
import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Iterator

import numpy as np
from tqdm.asyncio import tqdm
from backend_request_func import (async_request,
                                  RequestFuncInput,
                                  RequestFuncOutput)
from argparse import ArgumentParser
from random_prompt import RandomLmmPrompt

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    p99_latency_ms: float


def sample_random_requests(
        images_dir: str,
        num_prompts: int) -> Iterator[Tuple[str, str]]:
    generator = RandomLmmPrompt(images_dir)
    return generator(num_prompts)


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, str], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = 1.0 / request_rate
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    num_prompts: int,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    completed = 0
    latencys: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(outputs[i].generated_text)
            actual_output_lens.append(output_len)
            completed += 1
            latencys.append(outputs[i].latency)
        else:
            print(outputs[i].error)
            actual_output_lens.append(0)
            # latencys.append(outputs[i].latency)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=num_prompts,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=num_prompts / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_latency_ms=np.mean(latencys or 0) * 1000,
        median_latency_ms=np.median(latencys or 0) * 1000,
        std_latency_ms=np.std(latencys or 0) * 1000,
        p99_latency_ms=np.percentile(latencys or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    api_url: str,
    input_requests: Iterator[Tuple[str, str]],
    request_rate: float,
    num_prompts: int,
):
    print("Starting initial single prompt test run...")
    task_prompt, image_base64 = next(input_requests)
    test_input = RequestFuncInput(
        prompt=task_prompt,
        image=image_base64,
        api_url=api_url,
    )
    test_output = await async_request(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print(test_output.generated_text)
        print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: {request_rate} / sec")

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for task_prompt, image_base64 in tqdm(get_request(input_requests, request_rate), total=num_prompts):
        request_func_input = RequestFuncInput(
            prompt=task_prompt,
            image=image_base64,
            api_url=api_url,
        )
        tasks.append(
            asyncio.create_task(async_request(request_func_input=request_func_input))
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        num_prompts=num_prompts,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Latency', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Std latency (ms):",
                                    metrics.std_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 latency (ms):", metrics.p99_latency_ms))

    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_latency_ms": metrics.mean_latency_ms,
        "median_latency_ms": metrics.median_latency_ms,
        "std_latency_ms": metrics.std_latency_ms,
        "p99_latency_ms": metrics.p99_latency_ms,
        "output_lens": actual_output_lens,
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result


def main(args: argparse.Namespace):
    print(args)

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"


    if args.images_dir is not None:
        input_requests = sample_random_requests(
            images_dir=args.images_dir,
            num_prompts=args.num_prompts+1,
        )
    else:
        raise ValueError(f"Unknown images_dir: {args.images_dir}")

    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            input_requests=input_requests,
            request_rate=args.request_rate,
            num_prompts=args.num_prompts,
        )
    )

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        file_name = f"florence2-{args.request_rate}qps-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/",
        help="API endpoint.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Path of the image dataset to benchmark on.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=10,
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )

    args = parser.parse_args()
    main(args)
