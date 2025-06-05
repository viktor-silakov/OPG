"""Memory-optimized model initialization for Fish Speech"""

import torch
from loguru import logger
from fish_speech.models.text2semantic.llama import BaseTransformer, DualARTransformer, NaiveTransformer


def init_model_optimized(checkpoint_path, device, precision, compile=False, is_agent=False):
    """
    Memory-optimized model initialization
    - Automatically uses half precision on MPS for memory efficiency
    - Applies memory optimizations for Apple Silicon
    """
    model = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    # Memory optimization: Use best precision for device
    if device == "mps":
        if precision == torch.bfloat16:
            # Apple Silicon MPS works better with float16 for memory efficiency
            precision = torch.half
            logger.info("Using float16 for MPS memory optimization")
        elif precision == torch.float32:
            # Force conversion to avoid excessive RAM usage
            precision = torch.half
            logger.info("Converting float32 to float16 for MPS memory optimization")

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint with precision {precision}")

    if isinstance(model, DualARTransformer):
        from fish_speech.models.text2semantic.inference import (
            decode_one_token_ar_agent, decode_one_token_ar
        )
        decode_one_token = (
            decode_one_token_ar_agent if is_agent else decode_one_token_ar
        )
        logger.info("Using DualARTransformer")
    else:
        from fish_speech.models.text2semantic.inference import (
            decode_one_token_naive_agent, decode_one_token_naive
        )
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
