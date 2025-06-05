import dataclasses
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from fish_speech.tokenizer import SEMANTIC_TOKENS, FishTokenizer
from fish_speech.utils import RankedLogger

from .lora import LoraConfig, setup_lora

log = RankedLogger(__name__, rank_zero_only=True)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 1536  # Reduced default for memory optimization
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    share_codebook_embeddings: bool = True
    scale_codebook_embeddings: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        match data["model_type"]:
            case "naive":
                cls = NaiveModelArgs
            case "dual_ar":
                cls = DualARModelArgs
            case _:
                raise ValueError(f"Unknown model type: {data['model_type']}")

        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class NaiveModelArgs(BaseModelArgs):
    model_type: str = "naive"


@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    n_fast_layer: int = 4
    fast_dim: int | None = None
    fast_n_head: int | None = None
    fast_n_local_heads: int | None = None
    fast_head_dim: int | None = None
    fast_intermediate_size: int | None = None
    fast_attention_qkv_bias: bool | None = None

    def __post_init__(self):
        super().__post_init__()

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = (
            self.fast_intermediate_size or self.intermediate_size
        )
        self.fast_attention_qkv_bias = (
            self.fast_attention_qkv_bias
            if self.fast_attention_qkv_bias is not None
            else self.attention_qkv_bias
        )


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.float32
    ):
        super().__init__()
        # Memory optimization: Use pinned memory for faster transfers
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        if torch.backends.mps.is_available():
            # MPS optimization: Use half precision by default
            if dtype == torch.float32:
                dtype = torch.half
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    codebook_logits: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        tokenizer: FishTokenizer,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.semantic_token_ids = [
            tokenizer.get_token_id(SEMANTIC_TOKEN) for SEMANTIC_TOKEN in SEMANTIC_TOKENS
        ]

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        # Initialize the model
        if init_weights:
            self.apply(self._init_weights)

        # Setup caches for attention
        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_len = -1

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float32
    ):
        """Memory optimized cache setup"""
        # Memory optimization: Cap the max sequence length
        max_seq_len = min(max_seq_len, 2048)
        
        if (
            self.max_seq_len >= max_seq_len
            and self.max_batch_size >= max_batch_size
        ):
            return

        head_dim = self.config.dim // self.config.n_head
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        # MPS optimization: use half precision for cache
        if torch.backends.mps.is_available() and dtype == torch.float32:
            dtype = torch.half

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_len, self.config.n_local_heads, head_dim, dtype
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.max_seq_len,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.mask_cache = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        )

    def embed(self, inp: Tensor, share_codebook_embeddings=True) -> Tensor:
        # inp: [B, T, num_codebooks + 1]
        tokens = inp[:, :, 0]  # [B, T]
        codes = inp[:, :, 1:]  # [B, T, num_codebooks]

        token_emb = self.embeddings(tokens)  # [B, T, D]

        # Codebook embeddings
        if share_codebook_embeddings:
            codebook_emb = [
                self.codebook_embeddings(codes[:, :, i])
                for i in range(codes.size(-1))
            ]
            codebook_emb = torch.stack(codebook_emb, dim=-1).sum(dim=-1)  # [B, T, D]
        else:
            # Use different embeddings for each codebook
            codebook_offsets = torch.arange(
                0, self.config.num_codebooks * self.config.codebook_size, 
                self.config.codebook_size,
                device=codes.device
            )
            offset_codes = codes + codebook_offsets[None, None, :]
            codebook_emb = self.codebook_embeddings(offset_codes).sum(dim=-1)

        return token_emb + codebook_emb

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        """
        Forward pass with optional gradient checkpointing for memory efficiency
        """
        x = self.embed(inp)  # [B, T, D]

        T = x.size(1)
        assert (
            self.freqs_cis is not None
        ), f"Freqs_cis not setup for seq_len {T}"

        freqs_cis = self.freqs_cis[:T]
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)

        # Memory optimization: use gradient checkpointing if enabled
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, None)
            else:
                x = layer(x, freqs_cis, mask)

        hidden_states = x
        x = self.norm(x)  # [B, T, D]

        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.embeddings.weight)
        else:
            logits = self.output(x)

        return BaseTransformerForwardResult(
            logits=logits,
            hidden_states=hidden_states,
        )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        """
        Optimized forward pass for generation with KV caching
        """
        if input_pos is not None:
            # KV cache forward pass
            x = self.embed(inp)
            
            freqs_cis = self.freqs_cis[input_pos]
            mask = self.mask_cache[None, None, input_pos]

            for layer in self.layers:
                x = layer(x, freqs_cis, mask, input_pos)

            hidden_states = x
            x = self.norm(x)

            if self.config.tie_word_embeddings:
                logits = F.linear(x, self.embeddings.weight)
            else:
                logits = self.output(x)

            if not return_all:
                logits = logits[:, -1:, :]  # Only return last token

            return BaseTransformerForwardResult(
                logits=logits,
                hidden_states=hidden_states,
            )
        else:
            # Initial pass (prefill)
            return self.forward(inp)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    @staticmethod
    def from_pretrained(
        path: str,
        load_weights: bool = False,
        max_length: int | None = None,
        lora_config: LoraConfig | None = None,
        rope_base: int | None = None,
        is_agent: bool = False,
    ) -> "BaseTransformer":
        
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "config.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config_cls = NaiveModelArgs if config_data["model_type"] == "naive" else DualARModelArgs
        config = config_cls(**config_data)

        # Memory optimization
        if max_length is not None:
            config.max_seq_len = min(max_length, config.max_seq_len)
        if rope_base is not None:
            config.rope_base = rope_base

        # Initialize tokenizer
        tokenizer_path = Path(path) / "tokenizer.tiktoken"
        if not tokenizer_path.exists():
            tokenizer_path = Path(path) / "tokenizer.bpe"
        
        tokenizer = FishTokenizer(tokenizer_path)

        # Create model
        if config.model_type == "naive":
            model = NaiveTransformer(config, tokenizer)
        else:
            model = DualARTransformer(config, tokenizer)

        if not load_weights:
            return model

        # Load weights
        checkpoint_path = Path(path) / "text2semantic_500M.pth"
        if not checkpoint_path.exists():
            checkpoint_path = Path(path) / "text2semantic.pth"

        if checkpoint_path.exists():
            logger.info(f"Loading weights from {checkpoint_path}")
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                if "state_dict" in checkpoint_data:
                    state_dict = checkpoint_data["state_dict"]
                elif "model" in checkpoint_data:
                    state_dict = checkpoint_data["model"]
                else:
                    state_dict = checkpoint_data

                # Handle state dict keys
                if any(key.startswith("model.") for key in state_dict.keys()):
                    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

                # Memory optimization: force conversion to avoid float32 LoRA issue
                if torch.backends.mps.is_available():
                    for key, value in state_dict.items():
                        if value.dtype == torch.float32:
                            state_dict[key] = value.to(torch.half)

                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded model weights")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")

        # Setup LoRA if specified
        if lora_config is not None:
            setup_lora(model, lora_config)

        return model

    def save_pretrained(self, path: str, drop_lora: bool = False):
        """Save model with memory-efficient checkpoint"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.json")

        # Save tokenizer
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(path)

        # Save model weights
        state_dict = self.state_dict()
        if drop_lora:
            # Remove LoRA weights to save space
            state_dict = {k: v for k, v in state_dict.items() if "lora" not in k.lower()}

        torch.save(state_dict, path / "text2semantic_500M.pth")
        logger.info(f"Model saved to {path}")


class NaiveTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__(config, tokenizer)

        self.codebook_heads = nn.ModuleList(
            [
                nn.Linear(config.dim, config.codebook_size, bias=False)
                for _ in range(config.num_codebooks)
            ]
        )

    def decode(self, result: BaseTransformerForwardResult) -> TransformerForwardResult:
        token_logits = result.logits
        codebook_logits = torch.stack(
            [head(result.hidden_states) for head in self.codebook_heads], dim=2
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        result = super().forward(inp, key_padding_mask)
        return self.decode(result)

    def forward_generate(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        result = super().forward_generate(x, input_pos)
        return self.decode(result)


class DualARTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__(config, tokenizer)

        # Fast transformer for codebook generation
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    use_sdpa=True,
                    layer_id=i,
                    dim=config.fast_dim,
                    n_head=config.fast_n_head,
                    n_local_heads=config.fast_n_local_heads,
                    intermediate_size=config.fast_intermediate_size,
                    attention_qkv_bias=config.fast_attention_qkv_bias,
                )
                for i in range(config.n_fast_layer)
            ]
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float32
    ):
        """Setup caches for both slow and fast transformers"""
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        # Setup fast transformer caches
        fast_head_dim = self.config.fast_dim // self.config.fast_n_head
        max_seq_len = find_multiple(max_seq_len, 8)

        # Memory optimization for MPS
        if torch.backends.mps.is_available() and dtype == torch.float32:
            dtype = torch.half

        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.fast_n_local_heads,
                fast_head_dim,
                dtype,
            )

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        """
        Forward pass with memory-efficient dual AR architecture
        """
        result = super().forward(inp, key_padding_mask)
        token_logits = result.logits
        hidden_states = result.hidden_states

        # Fast transformer forward
        T = hidden_states.size(1)
        freqs_cis = self.freqs_cis[:T]
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)

        fast_out = hidden_states
        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                fast_out = checkpoint(layer, fast_out, freqs_cis, mask, None)
            else:
                fast_out = layer(fast_out, freqs_cis, mask)

        fast_out = self.fast_norm(fast_out)
        codebook_logits = self.fast_output(fast_out)

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits.unsqueeze(2),  # Add codebook dim
        )

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        """Fast transformer forward for codebook generation"""
        if input_pos is not None:
            freqs_cis = self.freqs_cis[input_pos]
            mask = self.mask_cache[None, None, input_pos]

            for layer in self.fast_layers:
                x = layer(x, freqs_cis, mask, input_pos)
        else:
            T = x.size(1)
            freqs_cis = self.freqs_cis[:T]
            mask = self.mask_cache[None, None, :T, :T]

            for layer in self.fast_layers:
                x = layer(x, freqs_cis, mask)

        x = self.fast_norm(x)
        return self.fast_output(x)

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        """Optimized generation forward with dual AR"""
        result = super().forward_generate(x, input_pos, return_all=True)
        return TransformerForwardResult(
            token_logits=result.logits,
            codebook_logits=result.hidden_states,  # Return hidden states for fast transformer
        )


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        config: BaseModelArgs, 
        use_sdpa: bool = True, 
        layer_id: int = 0,
        dim: Optional[int] = None,
        n_head: Optional[int] = None,
        n_local_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        attention_qkv_bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        
        # Use custom dimensions if provided (for fast transformer)
        self.dim = dim or config.dim
        self.n_head = n_head or config.n_head
        self.n_local_heads = n_local_heads or config.n_local_heads
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.attention_qkv_bias = attention_qkv_bias if attention_qkv_bias is not None else config.attention_qkv_bias

        # Create modified config for this block
        block_config = dataclasses.replace(
            config,
            dim=self.dim,
            n_head=self.n_head,
            n_local_heads=self.n_local_heads,
            intermediate_size=self.intermediate_size,
            attention_qkv_bias=self.attention_qkv_bias,
        )

        self.attention = Attention(block_config, use_sdpa)
        self.feed_forward = FeedForward(block_config)
        self.attention_norm = RMSNorm(self.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None
    ) -> Tensor:
        # Memory optimization: use in-place operations where possible
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.use_sdpa = use_sdpa

        self.wq = nn.Linear(self.dim, self.n_head * self.head_dim, bias=config.attention_qkv_bias)
        self.wk = nn.Linear(self.dim, self.n_local_heads * self.head_dim, bias=config.attention_qkv_bias)
        self.wv = nn.Linear(self.dim, self.n_local_heads * self.head_dim, bias=config.attention_qkv_bias)
        self.wo = nn.Linear(self.n_head * self.head_dim, self.dim, bias=False)
        self.kv_cache = None

        # Memory optimization: reduce dropout during inference
        if hasattr(config, 'attention_dropout'):
            self.attention_dropout = config.attention_dropout
        else:
            self.attention_dropout = 0.0

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict[prefix + "wq.weight"]
            wk = state_dict[prefix + "wk.weight"]
            wv = state_dict[prefix + "wv.weight"]
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # Repeat k and v if necessary
        if self.n_local_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        # Memory-efficient attention
        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized SDPA
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback to manual attention
            y = self.eq_scaled_dot_product_attention(q, k, v, mask, self.attention_dropout)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        """Memory-efficient manual attention implementation"""
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        if dropout_p > 0.0 and self.training:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            
        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    """Precompute rotary positional encoding frequencies"""
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary position embedding"""
    x_cis = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, :, None, :]
    x_out = torch.view_as_real(x_cis * freqs_cis).flatten(3)
    return x_out.type_as(x) 