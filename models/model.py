"""Encoder-decoder transformer with RoPE, SwiGLU and a KV-cached greedy decoder."""
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 8
    n_dec_layers: int = 8
    max_len: int = 512  # size of the RoPE tables; must cover the longest encoder/decoder sequence
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    @property
    def ffn_hidden(self):
        # ~8/3 * d_model rounded to a multiple of 64: same params/FLOPs as a 4 * d_model MLP.
        return int(round(8 * self.d_model / 3 / 64)) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize in fp32 so fp16 activations don't overflow in the mean of squares.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).to(x.dtype)


class Rotary(nn.Module):
    """Precomputed fp32 cos/sin tables, applied with the rotate-half convention."""

    def __init__(self, head_dim, max_len, base=10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, positions):
        """x: (B, H, T, hd); positions: (T,) long. Rotation is done in fp32."""
        cos = self.cos[positions]
        sin = self.sin[positions]
        xf = x.float()
        x1, x2 = xf.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return (xf * cos + rotated * sin).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).reshape(B, T, -1)

    def forward(self, x, rope, attn_mask=None, is_causal=False):
        q, k, v = (self._heads(t) for t in self.qkv(x).chunk(3, dim=-1))
        pos = torch.arange(x.size(1), device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return self.out(self._merge(y))

    def forward_step(self, x, rope, pos, k_cache, v_cache):
        """One decoding step. x: (B, 1, d). Writes the new key/value at `pos` into the
        preallocated caches (B, H, max_len, hd) and attends over positions [0, pos]."""
        q, k, v = (self._heads(t) for t in self.qkv(x).chunk(3, dim=-1))
        p = torch.tensor([pos], device=x.device)
        q, k = rope(q, p), rope(k, p)  # keys are rotated once, before they enter the cache
        k_cache[:, :, pos:pos + 1] = k
        v_cache[:, :, pos:pos + 1] = v
        y = F.scaled_dot_product_attention(q, k_cache[:, :, :pos + 1], v_cache[:, :, :pos + 1])
        return self.out(self._merge(y))


class CrossAttention(nn.Module):
    """No RoPE: query and key positions live in different sequences."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def memory_kv(self, memory):
        k, v = self.kv(memory).chunk(2, dim=-1)
        return self._heads(k), self._heads(v)

    def forward(self, x, memory_kv, attn_mask):
        k, v = memory_kv
        q = self._heads(self.q(x))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        B, _, T, _ = y.shape
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)  # fused W_gate | W_up
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), SelfAttention(cfg)
        self.norm2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, rope, attn_mask):
        x = x + self.attn(self.norm1(x), rope, attn_mask=attn_mask)
        return x + self.ffn(self.norm2(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.self_attn = RMSNorm(cfg.d_model), SelfAttention(cfg)
        self.norm2, self.cross_attn = RMSNorm(cfg.d_model), CrossAttention(cfg)
        self.norm3, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, rope, memory_kv, cross_mask):
        # Right padding: causal masking alone keeps real queries away from padded keys.
        x = x + self.self_attn(self.norm1(x), rope, is_causal=True)
        x = x + self.cross_attn(self.norm2(x), memory_kv, cross_mask)
        return x + self.ffn(self.norm3(x))

    def forward_step(self, x, rope, pos, k_cache, v_cache, memory_kv, cross_mask):
        x = x + self.self_attn.forward_step(self.norm1(x), rope, pos, k_cache, v_cache)
        x = x + self.cross_attn(self.norm2(x), memory_kv, cross_mask)
        return x + self.ffn(self.norm3(x))


def key_mask(padding_mask):
    """(B, S) bool, True = padding -> SDPA boolean mask (B, 1, 1, S), True = attend."""
    return ~padding_mask[:, None, None, :]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # One shared vocabulary: a single embedding for encoder, decoder and the output head.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rotary(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.enc_layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_enc_layers))
        self.dec_layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_dec_layers))
        self.enc_norm = RMSNorm(cfg.d_model)
        self.dec_norm = RMSNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, encoder_input_ids, encoder_padding_mask):
        x = self.embed(encoder_input_ids)
        mask = key_mask(encoder_padding_mask)
        for layer in self.enc_layers:
            x = layer(x, self.rope, mask)
        return self.enc_norm(x)

    def logits(self, h):
        return F.linear(h, self.embed.weight)

    def decode(self, decoder_input_ids, memory, encoder_padding_mask):
        x = self.embed(decoder_input_ids)
        mask = key_mask(encoder_padding_mask)
        for layer in self.dec_layers:
            x = layer(x, self.rope, layer.cross_attn.memory_kv(memory), mask)
        return self.logits(self.dec_norm(x))

    def forward(self, encoder_input_ids, encoder_padding_mask, decoder_input_ids):
        memory = self.encode(encoder_input_ids, encoder_padding_mask)
        return self.decode(decoder_input_ids, memory, encoder_padding_mask)


def token_loss(logits, labels):
    """Sum of cross-entropy over target tokens (labels == -100 ignored) and their count."""
    loss_sum = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                               ignore_index=-100, reduction="sum")
    return loss_sum, (labels != -100).sum()


# ---------------------------------------------------------------------------- generation

@torch.no_grad()
def greedy_generate(model, encoder_input_ids, encoder_padding_mask, max_new_tokens,
                    use_cache=True, deadline=None, compact=True):
    """Greedy decoding for one batch. Returns a list of token-id lists (without <bos>/<eos>).

    deadline: absolute time.monotonic() value; when passed, decoding stops and the tokens
    produced so far are returned. compact: drop finished rows from the batch (cache only).
    """
    cfg = model.cfg
    B = encoder_input_ids.size(0)
    device = encoder_input_ids.device
    assert max_new_tokens + 1 <= cfg.max_len

    memory = model.encode(encoder_input_ids, encoder_padding_mask)
    cross_mask = key_mask(encoder_padding_mask)
    out = torch.full((B, max_new_tokens), cfg.pad_id, dtype=torch.long, device=device)
    n_out = torch.zeros(B, dtype=torch.long, device=device)  # tokens produced before <eos>

    def record(rows, finished, nxt, pos):
        """Store the step's tokens for rows that are still generating; returns new finished."""
        live = ~finished
        out[rows[live], pos] = nxt[live]
        n_out[rows[live & (nxt != cfg.eos_id)]] += 1
        return finished | (nxt == cfg.eos_id)

    rows = torch.arange(B, device=device)  # original row index of each active row
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    if use_cache:
        # Cross-attention K/V: computed once per question for each decoder layer.
        mem_kv = [layer.cross_attn.memory_kv(memory) for layer in model.dec_layers]
        H, hd = cfg.n_heads, cfg.d_model // cfg.n_heads
        # Preallocated self-attention cache; positions 0..max_new_tokens-1 are ever fed in.
        # Under autocast the keys/values come out of the projections in the autocast dtype,
        # while memory (the residual stream) stays fp32.
        kv_dtype = (torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type)
                    else memory.dtype)
        k_cache = torch.zeros(len(model.dec_layers), B, H, max_new_tokens, hd, dtype=kv_dtype, device=device)
        v_cache = torch.zeros_like(k_cache)
        token = torch.full((B, 1), cfg.bos_id, dtype=torch.long, device=device)
        for pos in range(max_new_tokens):
            if deadline is not None and time.monotonic() > deadline:
                break
            x = model.embed(token)
            for i, layer in enumerate(model.dec_layers):
                x = layer.forward_step(x, model.rope, pos, k_cache[i], v_cache[i], mem_kv[i], cross_mask)
            nxt = model.logits(model.dec_norm(x))[:, -1].argmax(-1)
            finished = record(rows, finished, nxt, pos)
            if finished.all():
                break
            # Drop finished rows once they are a quarter of the batch, so later steps run on
            # fewer rows without copying the cache every time a single row ends.
            if compact and int(finished.sum()) * 4 >= finished.numel():
                keep = ~finished
                rows, nxt, finished, cross_mask = rows[keep], nxt[keep], finished[keep], cross_mask[keep]
                k_cache, v_cache = k_cache[:, keep], v_cache[:, keep]
                mem_kv = [(k[keep], v[keep]) for k, v in mem_kv]
            token = nxt[:, None]
    else:
        dec = torch.full((B, 1), cfg.bos_id, dtype=torch.long, device=device)
        for pos in range(max_new_tokens):
            if deadline is not None and time.monotonic() > deadline:
                break
            nxt = model.decode(dec, memory, encoder_padding_mask)[:, -1].argmax(-1)
            finished = record(rows, finished, nxt, pos)
            if finished.all():
                break
            dec = torch.cat([dec, nxt[:, None]], dim=1)

    out, n_out = out.tolist(), n_out.tolist()
    return [row[:n] for row, n in zip(out, n_out)]


@torch.no_grad()
def beam_generate(model, encoder_input_ids, encoder_padding_mask, max_new_tokens, num_beams=4,
                  length_penalty=1.0, temperature=1.0, do_sample=False, generator=None,
                  deadline=None, compact=True):
    """KV-cached beam search for one batch. Returns, per question, the token ids (without
    <bos>/<eos>) of the hypothesis with the best sum of log-probs / length ** length_penalty,
    where length counts the <eos>.

    Each step expands a question's K live beams into its 2K best continuations: those ending in
    <eos> and ranked within the top K go to the question's K finished slots, and the K best others
    stay live. A question is done when its finished slots are full and its best live beam, scored
    at the current length, can't beat the worst of them (the usual heuristic). If the deadline or
    max_new_tokens cuts a question short, its best live beam competes with the finished ones.
    With num_beams=1 and length_penalty=0 this is greedy decoding.

    temperature divides the logits before the log-softmax: below 1 it sharpens each step's
    distribution, so confident steps weigh more in a beam's score than uncertain ones.

    do_sample (beam sampling): the 2K continuations are drawn without replacement, each with
    probability proportional to its extended beam's probability, instead of taken as the top 2K;
    they are then ranked by score as usual. Uses the Gumbel-top-k trick: the top 2K of
    score + Gumbel noise is such a sample. Lower temperature -> closer to plain beam search.
    """
    cfg = model.cfg
    B, K, V, T = encoder_input_ids.size(0), num_beams, cfg.vocab_size, max_new_tokens
    device = encoder_input_ids.device
    assert max_new_tokens + 1 <= cfg.max_len
    neg_inf = float("-inf")

    memory = model.encode(encoder_input_ids, encoder_padding_mask)
    # Beam k of question b is row b*K + k; all beams of a question share its encoder output.
    cross_mask = key_mask(encoder_padding_mask).repeat_interleave(K, 0)
    mem_kv = [tuple(t.repeat_interleave(K, 0) for t in layer.cross_attn.memory_kv(memory))
              for layer in model.dec_layers]
    H, hd = cfg.n_heads, cfg.d_model // cfg.n_heads
    kv_dtype = (torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type)
                else memory.dtype)
    k_cache = torch.zeros(len(model.dec_layers), B * K, H, T, hd, dtype=kv_dtype, device=device)
    v_cache = torch.zeros_like(k_cache)

    # Finished hypotheses, indexed by original question; best first.
    fin_scores = torch.full((B, K), neg_inf, device=device)
    fin_seqs = torch.full((B, K, T), cfg.pad_id, dtype=torch.long, device=device)
    fin_lens = torch.zeros(B, K, dtype=torch.long, device=device)
    # Live beams of the active questions; qs maps them to original questions.
    qs = torch.arange(B, device=device)
    seqs = torch.full((B, K, T), cfg.pad_id, dtype=torch.long, device=device)
    scores = torch.full((B, K), neg_inf, device=device)
    scores[:, 0] = 0.0  # all beams start equal: expand only the first, or they'd be K copies
    done = torch.zeros(B, dtype=torch.bool, device=device)
    rank = torch.arange(2 * K, device=device)
    token = torch.full((B * K, 1), cfg.bos_id, dtype=torch.long, device=device)
    steps = 0  # tokens in each live beam

    for pos in range(T):
        if deadline is not None and time.monotonic() > deadline:
            break
        x = model.embed(token)
        for i, layer in enumerate(model.dec_layers):
            x = layer.forward_step(x, model.rope, pos, k_cache[i], v_cache[i], mem_kv[i], cross_mask)
        logp = F.log_softmax(model.logits(model.dec_norm(x))[:, -1].float() / temperature, dim=-1)
        n = len(qs)
        cand = (scores[:, :, None] + logp.view(n, K, V)).view(n, K * V)
        if do_sample:
            u = torch.rand(cand.shape, generator=generator, device=device).clamp_(min=1e-20)
            idx = (cand - torch.log(-torch.log(u))).topk(2 * K, dim=-1).indices
            vals, order = cand.gather(1, idx).sort(dim=-1, descending=True)
            idx = idx.gather(1, order)
        else:
            vals, idx = cand.topk(2 * K, dim=-1)  # sorted, best first
        beam, tok = idx // V, idx % V
        is_eos = tok == cfg.eos_id

        # <eos> continuations ranked within the top K become finished hypotheses.
        new_fin = is_eos & (rank < K) & torch.isfinite(vals) & ~done[:, None]
        fin_cand = torch.where(new_fin, vals / (pos + 1) ** length_penalty, neg_inf)
        cand_seqs = seqs.gather(1, beam[:, :, None].expand(-1, -1, T))  # tokens before the <eos>
        best, top = torch.cat([fin_scores[qs], fin_cand], 1).topk(K, dim=-1)
        fin_seqs[qs] = torch.cat([fin_seqs[qs], cand_seqs], 1).gather(1, top[:, :, None].expand(-1, -1, T))
        fin_lens[qs] = torch.cat([fin_lens[qs], torch.full_like(beam, pos)], 1).gather(1, top)
        fin_scores[qs] = best

        # The K best continuations that don't end stay live (each beam has one <eos>, so there are
        # at least K). Reorder the cache rows to follow each new beam's source beam.
        scores, sel = vals.masked_fill(is_eos, neg_inf).topk(K, dim=-1)
        src_beam, new_tok = beam.gather(1, sel), tok.gather(1, sel)
        seqs = cand_seqs.gather(1, sel[:, :, None].expand(-1, -1, T))
        seqs[:, :, pos] = new_tok
        steps = pos + 1
        src = (torch.arange(n, device=device)[:, None] * K + src_beam).view(-1)
        k_cache[:, :, :, :pos + 1] = k_cache[:, src, :, :pos + 1]
        v_cache[:, :, :, :pos + 1] = v_cache[:, src, :, :pos + 1]
        token = new_tok.view(-1, 1)

        worst_fin = fin_scores[qs, -1]
        done |= torch.isfinite(worst_fin) & (worst_fin >= scores[:, 0] / steps ** length_penalty)
        if done.all():
            break
        # As in greedy_generate: drop done questions once they are a quarter of the batch.
        if compact and int(done.sum()) * 4 >= n:
            keep = ~done
            rows = keep.repeat_interleave(K)
            qs, seqs, scores, done = qs[keep], seqs[keep], scores[keep], done[keep]
            token, cross_mask = token[rows], cross_mask[rows]
            k_cache, v_cache = k_cache[:, rows], v_cache[:, rows]
            mem_kv = [(k[rows], v[rows]) for k, v in mem_kv]

    out_seqs, out_lens = fin_seqs[:, 0], fin_lens[:, 0]
    if steps:  # questions cut short: their best live beam competes with the finished ones
        live = ~done
        q = qs[live]
        better = scores[live, 0] / steps ** length_penalty > fin_scores[q, 0]
        out_seqs[q[better]] = seqs[live, 0][better]
        out_lens[q[better]] = steps
    out_seqs, out_lens = out_seqs.tolist(), out_lens.tolist()
    return [row[:n] for row, n in zip(out_seqs, out_lens)]
