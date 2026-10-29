"""
emoji_sentiment_lib.py
Shared code for the Emoji-based Sentiment Classification project.
The training notebook writes this file into the artifacts folder, and the
offline demo notebook imports it from there, so preprocessing and the model
definition are guaranteed to be identical in both places.
"""
import os
import re
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import emoji

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_WS = re.compile(r"\s+")
# skin-tone modifiers + variation selector -> stripped so 👍🏽 and 👍 share one embedding
_EMOJI_NOISE = re.compile("[\U0001F3FB-\U0001F3FF️]")


# ----------------------------------------------------------------------------
# Text preprocessing
# ----------------------------------------------------------------------------
def clean_text(text):
    """Replace links and @mentions with placeholders, squeeze whitespace."""
    text = _URL.sub("http", str(text))
    text = _USER.sub("@user", text)
    return _WS.sub(" ", text).strip()


def norm_emoji(e):
    """Canonical form of an emoji used as the vocabulary key."""
    out = _EMOJI_NOISE.sub("", e)
    return out if out else e


def extract_emojis(text):
    return [m["emoji"] for m in emoji.emoji_list(text)]


def has_emoji(text):
    return emoji.emoji_count(text) > 0


def _emoji_to_words(chars, data):
    name = (data or {}).get("en", ":emoji:")
    return " " + name.strip(":").replace("_", " ") + " "


def demojize_text(text):
    """'exam tomorrow 😭' -> 'exam tomorrow loudly crying face'"""
    return _WS.sub(" ", emoji.replace_emoji(text, replace=_emoji_to_words)).strip()


def emoji_only_text(text):
    """Keep only the emojis of a tweet: 'love this 😍🔥' -> '😍 🔥'"""
    return " ".join(extract_emojis(text))


def word_dropout(text, p, rng):
    """Randomly drop non-emoji words (training only) so the model learns to lean on emojis."""
    if p <= 0:
        return text
    toks = text.split()
    kept = [w for w in toks if has_emoji(w) or rng.random() >= p]
    return " ".join(kept) if kept else text


# ----------------------------------------------------------------------------
# Emoji vocabulary (+ lexicon prior from the Emoji Sentiment Ranking)
# ----------------------------------------------------------------------------
class EmojiVocab:
    PAD, UNK = 0, 1
    NEUTRAL_PRIOR = [1 / 3, 1 / 3, 1 / 3, 0.0]  # [p_neg, p_neu, p_pos, in_lexicon]

    def __init__(self, itos, prior):
        self.itos = list(itos)
        self.stoi = {e: i for i, e in enumerate(self.itos)}
        self.prior = np.asarray(prior, dtype=np.float32)

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, lexicon_df, train_texts, min_count=1):
        itos = ["<pad>", "<unk>"]
        prior = [cls.NEUTRAL_PRIOR, cls.NEUTRAL_PRIOR]
        seen = set(itos)
        if lexicon_df is not None and len(lexicon_df):
            for row in lexicon_df.itertuples(index=False):
                key = norm_emoji(str(row.emoji))
                if key in seen:
                    continue
                seen.add(key)
                itos.append(key)
                prior.append([float(row.p_neg), float(row.p_neu), float(row.p_pos), 1.0])
        counts = Counter(norm_emoji(e) for t in train_texts for e in extract_emojis(t))
        for key, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            if c >= min_count and key not in seen:
                seen.add(key)
                itos.append(key)
                prior.append(cls.NEUTRAL_PRIOR)
        return cls(itos, prior)

    def encode(self, text, max_emojis):
        ids = [self.stoi.get(norm_emoji(e), self.UNK) for e in extract_emojis(text)]
        return ids[:max_emojis]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "prior": self.prior.tolist()}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["itos"], d["prior"])


# ----------------------------------------------------------------------------
# Batch builders
# ----------------------------------------------------------------------------
def prepare_plain_batch(tokenizer, texts, mode, max_len):
    """mode='raw' -> emojis kept as characters (setting B); mode='demoji' -> emojis as words (setting C)."""
    texts = [clean_text(t) for t in texts]
    if mode == "demoji":
        texts = [demojize_text(t) for t in texts]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt", return_token_type_ids=False)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def prepare_gated_batch(tokenizer, vocab, texts, max_len, max_emojis, word_drop=0.0, rng=None):
    """Inputs for the two-branch model: tokenised demojized text + emoji ids."""
    texts = [clean_text(t) for t in texts]
    text_in = [word_dropout(t, word_drop, rng) if (word_drop > 0 and rng is not None) else t for t in texts]
    text_in = [demojize_text(t) for t in text_in]
    enc = tokenizer(text_in, padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt", return_token_type_ids=False)
    ids = [vocab.encode(t, max_emojis) for t in texts]
    width = max(1, max(len(x) for x in ids))
    emoji_ids = torch.zeros(len(ids), width, dtype=torch.long)
    emoji_mask = torch.zeros(len(ids), width, dtype=torch.float)
    for i, row in enumerate(ids):
        if row:
            emoji_ids[i, :len(row)] = torch.tensor(row)
            emoji_mask[i, :len(row)] = 1.0
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "emoji_ids": emoji_ids, "emoji_mask": emoji_mask}


# ----------------------------------------------------------------------------
# text branch + emoji branch + trust gate
# ----------------------------------------------------------------------------
class GatedEmojiSentimentModel(nn.Module):
    def __init__(self, encoder, emoji_prior, n_labels=3, emoji_dim=64, hid=256, gate_bias_init=2.0):
        super().__init__()
        self.encoder = encoder
        H = getattr(encoder.config, "hidden_size", None) or getattr(encoder.config, "dim", 768)
        self.register_buffer("emoji_prior", torch.as_tensor(np.asarray(emoji_prior), dtype=torch.float32))
        n_emojis, prior_dim = self.emoji_prior.shape

        # emoji branch
        self.emoji_emb = nn.Embedding(n_emojis, emoji_dim, padding_idx=0)
        self.emoji_proj = nn.Sequential(nn.Linear(emoji_dim + prior_dim, hid), nn.GELU(), nn.Dropout(0.1))
        self.emoji_attn = nn.Linear(hid, 1)
        self.empty_emoji = nn.Parameter(torch.zeros(hid))

        # text branch
        self.text_proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, hid), nn.GELU())

        # gate: starts at sigmoid(gate_bias_init) ~ 0.88 -> trusts emojis by default
        self.gate = nn.Linear(2 * hid, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)

        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hid, n_labels))
        self.emoji_head = nn.Linear(hid, n_labels)  # auxiliary head: emoji branch alone

    def forward(self, input_ids, attention_mask, emoji_ids, emoji_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        t = self.text_proj(h)

        x = torch.cat([self.emoji_emb(emoji_ids), self.emoji_prior[emoji_ids]], dim=-1)
        x = self.emoji_proj(x)                                    # [B, E, hid]
        scores = self.emoji_attn(x).squeeze(-1).float()
        scores = scores.masked_fill(emoji_mask == 0, -1e4)
        attn = torch.softmax(scores, dim=-1) * emoji_mask         # zero for padding / no-emoji rows
        e = (attn.unsqueeze(-1).to(x.dtype) * x).sum(1)

        has = (emoji_mask.sum(1, keepdim=True) > 0).to(e.dtype)   # [B, 1]
        e = has * e + (1 - has) * self.empty_emoji.to(e.dtype)
        t = t.to(e.dtype)

        g = torch.sigmoid(self.gate(torch.cat([t, e], dim=-1))) * has
        fused = g * e + (1 - g) * t
        return {"logits": self.classifier(fused), "emoji_logits": self.emoji_head(e),
                "gate": g.squeeze(-1), "attn": attn, "has_emoji": has.squeeze(-1)}


def load_gated_model(model_dir, device="cpu"):
    from transformers import AutoConfig, AutoModel
    with open(os.path.join(model_dir, "gated_config.json")) as f:
        meta = json.load(f)
    vocab = EmojiVocab.load(os.path.join(model_dir, "emoji_vocab.json"))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(model_dir, "encoder_config")))
    model = GatedEmojiSentimentModel(encoder, vocab.prior, n_labels=len(LABELS), emoji_dim=meta["emoji_dim"],
                                     hid=meta["hid"], gate_bias_init=meta["gate_bias_init"])
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval(), vocab, meta


# ----------------------------------------------------------------------------
# Offline predictor used by the demo notebook
# ----------------------------------------------------------------------------
class SentimentPredictor:
    MODEL_NAMES = {"B": "B: DistilBERT (raw emojis)",
                   "C": "C: DistilBERT (demojized)",
                   "D": "D: Gated emoji model (ours)"}

    def __init__(self, art_dir, device=None, which=("B", "C", "D")):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        mdir = os.path.join(art_dir, "models")
        self.tok = AutoTokenizer.from_pretrained(os.path.join(art_dir, "tokenizer"))
        self.models = {}
        if "B" in which:
            self.models["B"] = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(mdir, "B_raw")).to(self.device).eval()
        if "C" in which:
            self.models["C"] = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(mdir, "C_demoji")).to(self.device).eval()
        if "D" in which:
            self.models["D"], self.vocab, self.meta = load_gated_model(os.path.join(mdir, "D_gated"), self.device)
        self.max_len = getattr(self, "meta", {}).get("max_len", 128)
        self.max_emojis = getattr(self, "meta", {}).get("max_emojis", 8)

    @torch.no_grad()
    def predict(self, text):
        out = {}
        for key, mode in (("B", "raw"), ("C", "demoji")):
            if key in self.models:
                b = prepare_plain_batch(self.tok, [text], mode, self.max_len)
                b = {k: v.to(self.device) for k, v in b.items()}
                p = F.softmax(self.models[key](**b).logits.float(), -1)[0].cpu().numpy()
                out[key] = {"label": LABELS[int(p.argmax())], "probs": p}
        if "D" in self.models:
            b = prepare_gated_batch(self.tok, self.vocab, [text], self.max_len, self.max_emojis)
            b = {k: v.to(self.device) for k, v in b.items()}
            o = self.models["D"](**b)
            p = F.softmax(o["logits"].float(), -1)[0].cpu().numpy()
            ems = extract_emojis(clean_text(text))[: self.max_emojis]
            w = o["attn"][0, : len(ems)].float().cpu().numpy()
            out["D"] = {"label": LABELS[int(p.argmax())], "probs": p,
                        "gate": float(o["gate"][0]), "emoji_weights": list(zip(ems, w.tolist()))}
        return out
