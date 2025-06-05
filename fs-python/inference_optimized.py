import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech.models.text2semantic.llama import BaseModelArgs
from fish_speech.text import clean_text, split_text
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync_agent(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def sample_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs_agent(
        logits=logits[:, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync_agent(probs)
    return idx_next, probs


def decode_one_token_ar_agent(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # print(x, input_pos)
    x = model.forward_generate(x, input_pos)
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample_agent(
            logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample_agent(
            logits,
            previous_tokens=(
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~semantic_ids_tensor[:, None, None].expand(-1, codebooks.shape[1] - 1, 1),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks.view(1, -1)


def decode_one_token_naive_agent(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    assert input_pos.shape[-1] == 1, input_pos.shape
    x = model.forward_generate(x, input_pos)
    logits = x.token_logits
    codebook_logits = x.codebook_logits

    # Sample tokens
    token = sample_agent(logits, previous_tokens=None, **sampling_kwargs)[0]
    token = token.view(1, 1)

    codebooks = [token]
    for i in range(model.config.num_codebooks):
        a = sample_agent(
            codebook_logits[:, :, i, :],
            previous_tokens=(
                previous_tokens[:, i + 1] if previous_tokens is not None else None
            ),
            **sampling_kwargs,
        )[0]
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    # Mask out non-semantic tokens
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~semantic_ids_tensor[:, None, None].expand(-1, codebooks.shape[1] - 1, 1),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks.view(1, -1)


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # print(x, input_pos)
    x = model.forward_generate(x, input_pos)
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample(
            logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1] if previous_tokens is not None else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=0).unsqueeze(0)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~semantic_ids_tensor[:, None, None].expand(-1, codebooks.shape[1] - 1, 1),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks.view(1, -1)


def decode_one_token_naive(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    assert input_pos.shape[-1] == 1, input_pos.shape
    x = model.forward_generate(x, input_pos)
    logits = x.token_logits
    codebook_logits = x.codebook_logits

    # Sample tokens
    token = sample(logits, previous_tokens=None, **sampling_kwargs)[0]
    token = token.view(1, 1)

    codebooks = [token]
    for i in range(model.config.num_codebooks):
        a = sample(
            codebook_logits[:, :, i, :],
            previous_tokens=(
                previous_tokens[i + 1] if previous_tokens is not None else None
            ),
            **sampling_kwargs,
        )[0]
        codebooks.append(a)

    return torch.stack(codebooks, dim=0).unsqueeze(0).view(1, -1)


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    previous_tokens = cur_token[:, -(model.config.num_codebooks + 1) :]

    for i in tqdm(
        range(num_new_tokens), desc="Generating tokens", unit=" tokens"
    ):
        # If we have too many codebooks, we just use the first two
        if previous_tokens.size(-1) > (model.config.num_codebooks + 1) * 50:
            previous_tokens = previous_tokens[:, -((model.config.num_codebooks + 1) * 10) :]

        # Actually better not to restrict the previous tokens
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token = decode_one_token(
                model,
                cur_token[:, input_pos],
                torch.tensor([input_pos.shape[-1]], device=cur_token.device, dtype=torch.long),
                previous_tokens=None,
                **sampling_kwargs,
            )
            new_tokens.append(next_token.clone())
            cur_token = torch.cat([cur_token, next_token], dim=1)

        input_pos = input_pos[-1:] + 1

        # Memory optimization: limit the sliding window
        if cur_token.size(1) > 2048:  # Reduced max length
            # Keep only recent tokens
            cur_token = cur_token[:, -1024:]
            input_pos = torch.arange(1024, device=cur_token.device)

    return cur_token


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: NaiveTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Optimized generation function with memory management
    """
    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1, 
            max_seq_len=min(2048, model.config.max_seq_len),  # Reduced cache size
            dtype=dtype
        )

    # Prefill
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        if isinstance(model, DualARTransformer):
            result = model.forward_generate(prompt)
        else:
            result = model.forward_generate(prompt)

    seq_len = prompt.size(1)
    input_pos = torch.arange(seq_len, device=device)
    next_token = decode_one_token(
        model, prompt, input_pos, **sampling_kwargs
    ).clone()

    # Memory optimization: create smaller initial tensor
    T = seq_len + max_new_tokens
    if T > 2048:  # Cap max sequence length
        T = 2048
        max_new_tokens = min(max_new_tokens, T - seq_len)
    
    seq = torch.cat([prompt, next_token], dim=1)

    input_pos = torch.tensor([seq_len], device=device, dtype=torch.long)
    generated_tokens = decode_n_tokens(
        model,
        seq,
        input_pos,
        max_new_tokens - 1,
        [True] * model.config.num_codebooks,
        decode_one_token,
        **sampling_kwargs,
    )

    return generated_tokens


def decode_n_tokens_agent(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    previous_tokens = cur_token[:, -(model.config.num_codebooks + 1) :]

    im_end_rate = 0
    for i in tqdm(
        range(num_new_tokens), desc="Generating tokens", unit=" tokens"
    ):
        if previous_tokens.size(-1) > (model.config.num_codebooks + 1) * 50:
            previous_tokens = previous_tokens[:, -((model.config.num_codebooks + 1) * 10) :]

        # Actually better not to restrict the previous tokens
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token = decode_one_token(
                model,
                cur_token[:, input_pos],
                torch.tensor([input_pos.shape[-1]], device=cur_token.device, dtype=torch.long),
                semantic_ids,
                previous_tokens=None,
                **sampling_kwargs,
            )

            new_tokens.append(next_token.clone())
            cur_token = torch.cat([cur_token, next_token], dim=1)

        input_pos = input_pos[-1:] + 1

        # Check for early stopping
        is_im_end = next_token.view(-1)[0] == im_end_id
        im_end_rate = im_end_rate * 0.9 + 0.1 * is_im_end.float().item()

        if im_end_rate > early_stop_threshold:
            logger.info(f"Early stopping at step {i} with im_end_rate {im_end_rate}")
            break

        # Memory optimization
        if cur_token.size(1) > 2048:
            cur_token = cur_token[:, -1024:]
            input_pos = torch.arange(1024, device=cur_token.device)

    return cur_token


@torch.no_grad()
@torch.inference_mode()
def generate_agent(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    num_samples: int = 1,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=num_samples, 
            max_seq_len=min(2048, model.config.max_seq_len),  # Reduced cache size
            dtype=dtype
        )

    # Create the initial prompt
    prompt = prompt.repeat(num_samples, 1)

    # Prefill
    if isinstance(model, DualARTransformer):
        result = model.forward_generate(prompt)
    else:
        result = model.forward_generate(prompt)

    seq_len = prompt.size(1)
    input_pos = torch.arange(seq_len, device=device)
    next_token = decode_one_token(
        model, prompt, input_pos, semantic_ids, **sampling_kwargs
    ).clone()

    # Memory optimization
    T = seq_len + max_new_tokens
    if T > 2048:
        T = 2048
        max_new_tokens = min(max_new_tokens, T - seq_len)
    
    seq = torch.cat([prompt, next_token], dim=1)

    input_pos = torch.tensor([seq_len], device=device, dtype=torch.long)
    generated_tokens = decode_n_tokens_agent(
        model,
        seq,
        input_pos,
        max_new_tokens - 1,
        semantic_ids,
        im_end_id,
        decode_one_token,
        early_stop_threshold,
        **sampling_kwargs,
    )

    return generated_tokens


def encode_tokens(
    tokenizer,
    string,
    device="cuda",
    prompt_tokens=None,
    num_codebooks=4,
):
    if prompt_tokens is None:
        # Encode text
        string = clean_text(string)
        if isinstance(tokenizer, FishTokenizer):
            new_tokens = tokenizer.encode(
                string, allowed_special=set(), disallowed_special=()
            )[0]
        else:
            new_tokens = tokenizer.encode(string, add_special_tokens=False)

        # Add im_start and im_end
        tokens = [tokenizer.get_token_id("<|im_start|>")] + new_tokens + [tokenizer.get_token_id("<|im_end|>")]

        # Convert to tensor
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)

        # Add codebook placeholders  
        placeholder = torch.zeros(
            (1, tokens.size(-1), num_codebooks), device=device, dtype=torch.long
        )
        placeholder.fill_(CODEBOOK_PAD_TOKEN_ID)

        return torch.cat([tokens.unsqueeze(-1), placeholder], dim=-1).squeeze(0).t()
    else:
        tokens, codes = prompt_tokens[:1], prompt_tokens[1:]
        
        # Ensure we have the right shape
        if codes.shape[0] != num_codebooks:
            codes = codes.t()[:num_codebooks].t()
        
        result = torch.zeros((tokens.size(0) + codes.size(0), codes.size(1)), device=device, dtype=torch.long)
        result[0] = tokens.squeeze()
        result[1:] = codes
        
        return result


def init_model(checkpoint_path, device, precision, compile=False, is_agent=False):
    """
    Optimized model initialization with memory management
    """
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    # Memory optimization: Use best precision for device
    if device == "mps" and precision == torch.bfloat16:
        # Apple Silicon MPS works better with float16 for memory efficiency
        precision = torch.half
        logger.info("Using float16 for MPS memory optimization")
    elif device == "mps" and precision == torch.float32:
        # Force conversion to avoid excessive RAM usage
        precision = torch.half
        logger.info("Converting float32 to float16 for MPS memory optimization")

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint with precision {precision}")

    if isinstance(model, DualARTransformer):
        decode_one_token = (
            decode_one_token_ar_agent if is_agent else decode_one_token_ar
        )
        logger.info("Using DualARTransformer")
    else:
        decode_one_token = (
            decode_one_token_naive_agent if is_agent else decode_one_token_naive
        )
        logger.info("Using NaiveTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 1536,  # Reduced default max length
    chunk_length: int = 100,  # Reduced chunk length  
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    """
    Optimized long generation with aggressive memory management
    """
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    im_end_id = tokenizer.get_token_id("<|im_end|>")

    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    encoded_prompts = [
        Conversation(
            messages=[
                Message(
                    role="system",
                    parts=[TextPart(text="Speak out the provided text.")],
                    cal_loss=False,
                )
            ]
        )
        .encode_for_inference(
            tokenizer=tokenizer,
            num_codebooks=model.config.num_codebooks,
        )
        .to(device)
    ]

    if use_prompt:
        for idx, (t, c) in enumerate(zip(prompt_text, prompt_tokens)):
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=t,
                    device=device,
                    prompt_tokens=c,
                    num_codebooks=model.config.num_codebooks,
                )
            )

    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                device=device,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0

        while seg_idx < len(encoded):
            logger.info(
                f"Generating sentence {seg_idx + 1}/{len(encoded)} of sample {sample_idx + 1}/{num_samples}"
            )

            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last tokens with reduced window
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 512 - sum(  # Reduced buffer
                    t.shape[1] for t in encoded_prompts
                ):
                    break

            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list, always make sure first segment is included to avoid drift
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[:2] + global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            if use_prompt:
                partial_encoded = encoded_prompts + partial_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if sample_idx == 0 and seg_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            if torch.cuda.is_available():
                logger.info(
                    f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
                )

            # Put the generated tokens
            # since there is <im_end>, we remove last token
            codes = y[1:, prompt_length + 1 :].clone()

            # Memory cleanup
            del y
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            yield GenerateResponse(action="sample", codes=codes)

            seg_idx += 1

        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_one_token = threading.Event()

    def worker():
        model, decode_one_token = init_model(checkpoint_path, device, precision, compile)
        init_one_token.set()

        while True:
            req: GenerateRequest = input_queue.get()
            try:
                result = generate_long(
                    model=model,
                    device=device,
                    decode_one_token=decode_one_token,
                    **req.request,
                )
                for chunk in result:
                    req.response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                req.response_queue.put(
                    WrappedGenerateResponse(status="error", response=e)
                )

    threading.Thread(target=worker, daemon=True).start()
    return input_queue, init_one_token


def launch_thread_safe_queue_agent(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_one_token = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile, is_agent=True
        )
        init_one_token.set()

        while True:
            req: GenerateRequest = input_queue.get()
            try:
                result = generate_agent(
                    model=model,
                    decode_one_token=decode_one_token,
                    **req.request,
                )
                req.response_queue.put(
                    WrappedGenerateResponse(status="success", response=result)
                )
            except Exception as e:
                req.response_queue.put(
                    WrappedGenerateResponse(status="error", response=e)
                )

    threading.Thread(target=worker, daemon=True).start()
    return input_queue, init_one_token


# CLI remains the same but with optimized defaults
@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/fish-speech-1.5",
)
@click.option("--device", type=str, default="mps")  # Changed default to mps
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=True)  # Changed default to True
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=100)  # Reduced default
@click.option("--output-dir", type=Path, default="temp")
def main(
    text: str,
    prompt_text: Optional[list[str]],
    prompt_tokens: Optional[list[Path]],
    num_samples: int,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    assert device in [
        "cuda",
        "cpu",
        "mps",
    ], f"Device {device} is not supported"

    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS is not available, using CPU instead")
        device = "cpu"

    torch.manual_seed(seed)

    model, decode_one_token = init_model(
        checkpoint_path,
        device,
        precision=torch.half if half else torch.bfloat16,
        compile=compile,
    )

    # Memory optimization: ensure cache is properly sized
    if torch.backends.mps.is_available() and device == "mps":
        torch.mps.empty_cache()

    prompt_tokens_np = []
    if prompt_tokens:
        for prompt_token in prompt_tokens:
            prompt_tokens_np.append(torch.from_numpy(np.load(prompt_token)).to(device))

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens_np if prompt_tokens_np else None,
    )

    idx = 0
    for response in generator:
        if response.action == "sample":
            codes = response.codes.cpu().numpy()
            np.save(output_dir / f"codes_{idx}.npy", codes)
            logger.info(f"Saved codes to {output_dir / f'codes_{idx}.npy'}")
            idx += 1
            
            # Memory cleanup after each sample
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    logger.info("Generation completed")


if __name__ == "__main__":
    main() 