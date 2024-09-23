import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncInput:
    prompt: str
    image: str
    api_url: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    error: str = ""


async def async_request(
    request_func_input: RequestFuncInput,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "task_prompt": request_func_input.prompt,
            "image": request_func_input.image,
        }

        output = RequestFuncOutput()

        generated_text = ""
        start_time = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        generated_text = generated_text + chunk_bytes
                        if not chunk_bytes:
                            continue

                    output.generated_text = generated_text
                    output.success = True
                    end_time  = time.perf_counter()
                    output.latency = end_time - start_time
                else:
                    output.error = f"failed to post requests with error {response}"
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output
