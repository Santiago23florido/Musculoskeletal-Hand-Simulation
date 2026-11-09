"""P2: open-loop test of C4 architectures that see batch and history context.

P1 showed C4 ranks a unit's true candidates well (median AUC 0.985) but judges each
candidate alone; with ~50 candidates per true discharge the residual ranking errors
decide RoA. Data, calibration-lag labels, the 24 P1 inputs (22 L2 + C1 posture),
optimiser, epochs, seeds, epoch selection and TPE are P1's. Only the head changes:

- ``wide24``: a larger per-candidate MLP (capacity control, P1's i.i.d. loop);
- ``set24``: candidates of one unit attend to each other within the batch and to the
  previous batch, with a learned bias on their time separation;
- ``setrec24``: ``set24`` plus a per-unit GRU across batches that receives the posture
  trajectory and summaries of earlier candidates (history and movement context);
- ``mdiff24``: generative arm on the ``setrec24`` trunk, a masked discrete diffusion
  (discrete flow matching with a mask source) over the joint discharge labels of a
  unit-batch and the previous batch. Labels enter only the last attention block as
  key/value terms, never a candidate's own; decoding is batch by batch: the previous
  batch's decided labels, two confidence-ordered reveals, then a readout of every
  candidate given all other labels.

Every head is the prior at initialisation and is decided open loop on the sealed L2
final candidates, exactly as P1. Nothing is deployed or changes an accepted level.
"""

import argparse
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import OUTPUT, VALIDATION, now, records, register, sha256, stub, write_new
from .experiment import checked
from .memory_experiment import SEEDS, objective_value, read
from .posture_c4 import FOLDER as P1
from .posture_c4 import LAG_BATCHES, inputs, key, l2_paths
from .train_likelihood import EPOCHS, labelled, quick_scores

FOLDER = OUTPUT / "P2_context_c4"
ARMS = ("wide24", "set24", "setrec24", "mdiff24")
KINDS = dict(wide24="wide", set24="set", setrec24="setrec", mdiff24="mdiff")
BATCH = 204
CHUNK = 64  # at most 64 batches (6.4 s) per truncated-BPTT step; GRU state carried across
BUDGET = 16_000_000  # attention elements (units x batches x K x 2K x heads) per chunk
STEPS = 3  # generative arm: two reveal passes, then a readout given all other labels
KEEP = 5 * 2048  # candidates before 5 s are calibration: unlabelled, never revealed
EDGES = (1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 61, 91, 136, 204, 300)  # |dt| buckets, samples
CONTEXT_EPOCHS = 3  # P1 used 10; 3 over P1's 420 trajectories is 7x the pilot's updates
BOOTSTRAP = 2000


class BiasedAttention(nn.Module):
    """Pre-norm block: queries are the batch; keys are the previous batch and the batch."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)
        self.bias = nn.Embedding(len(EDGES) + 1, heads)
        nn.init.zeros_(self.bias.weight)
        self.feed = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, x, previous, bucket, valid, labels=None):
        # x [S, K, W]; previous [S, K, W] (static embeddings); bucket [S, K, 2K]; valid [S, 2K]
        # labels: label-state key and value terms [S, h, 2K, d]; a query never sees its own.
        s, k, w = x.shape
        h, d = self.heads, w // self.heads
        normed = self.norm(x)
        q = self.query(normed).view(s, k, h, d).transpose(1, 2)
        both = torch.cat([previous, normed], 1)
        key_, value = self.key_value(both).view(s, 2 * k, 2, h, d).permute(2, 0, 3, 1, 4)
        scores = q @ key_.transpose(-1, -2) / math.sqrt(d) + self.bias(bucket).permute(0, 3, 1, 2)
        if labels is not None:
            eye = torch.eye(k, dtype=torch.bool, device=x.device)
            own = torch.cat([torch.zeros_like(eye), eye], 1)  # query i, key K + i
            label_key, label_value = labels
            scores = scores + (q @ label_key.transpose(-1, -2) / math.sqrt(d)).masked_fill(own, 0)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        weights = torch.nan_to_num(scores.softmax(-1))  # sets without any token
        mixed = weights @ value
        if labels is not None:
            mixed = mixed + weights.masked_fill(own, 0) @ label_value
        x = x + self.output(mixed.transpose(1, 2).reshape(s, k, w))
        return x + self.feed(x)


class ContextLikelihood(nn.Module):
    """Residual log-odds over the Pareto prior, bounded by +-4 and zero at initialisation."""

    def __init__(self, kind, features=24, width=32, heads=4, layers=2, bound=4.0):
        super().__init__()
        if kind not in ("wide", "set", "setrec", "mdiff") or features != 24:
            raise ValueError("Undeclared C4 context head")
        self.spec = dict(
            kind=kind, features=features, width=width, heads=heads, layers=layers, bound=bound
        )
        self.register_buffer("edges", torch.tensor(EDGES, dtype=torch.int32), persistent=False)
        if kind == "wide":
            self.layers = nn.Sequential(
                nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
            )
            last = self.layers[-1]
        else:
            self.embed = nn.Sequential(
                nn.Linear(features + 1, width), nn.GELU(), nn.Linear(width, width)
            )
            self.blocks = nn.ModuleList(BiasedAttention(width, heads) for _ in range(layers))
            if kind in ("setrec", "mdiff"):
                self.gru = nn.GRU(2 * width + 3, width, batch_first=True)
                self.film = nn.Linear(width, 2 * width)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
            if kind == "mdiff":
                self.label = nn.Embedding(3, width)  # 0 masked, 1 rejected, 2 accepted
                self.label_kv = nn.Linear(width, 2 * width)
            self.head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
            )
            last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @property
    def recurrent(self):
        return self.spec["kind"] in ("setrec", "mdiff")

    def forward(self, chunk, hidden=None, history=True, labels=None):
        """Logits [m, C, K] for a chunk and the GRU state after its last batch.

        The context of batch c is the GRU state after batch c - 1, so history is strictly
        causal; batch c's own candidates enter only through attention.
        """
        x = chunk["x"]
        if self.spec["kind"] == "wide":
            prior = (chunk["primary"].float() * 2 - 1) * math.log(9)
            out = self.layers(x.clamp(-6, 6)).squeeze(-1)
            return prior + self.spec["bound"] * out.tanh(), hidden
        parts, hidden = self.trunk(chunk, hidden, history)
        return self.readout(parts, labels=labels), hidden

    def trunk(self, chunk, hidden, history):
        """Everything except the last attention block, for all batches of a chunk at once."""
        x, valid = chunk["x"], chunk["valid"]
        m, c, k, _ = x.shape
        e = self.embed(torch.cat([x.clamp(-6, 6), chunk["offset"][..., None]], -1))
        pe = self.embed(torch.cat([chunk["px"].clamp(-6, 6), chunk["poffset"][..., None]], -1))
        if self.recurrent:
            if hidden is None:
                hidden = torch.zeros(1, m, self.spec["width"], device=x.device)
            mask = valid[..., None].float()
            count = mask.sum(2)
            mean = (e * mask).sum(2) / count.clamp_min(1)
            peak = e.masked_fill(~valid[..., None], -1e4).amax(2) * (count > 0)
            summary = torch.cat([mean, peak, torch.log1p(count) / 4, chunk["posture"]], -1)
            states, _ = self.gru(summary, hidden)
            context = torch.cat([hidden.transpose(0, 1), states[:, :-1]], 1)
            if not history:
                context = torch.zeros_like(context)
            gamma, beta = self.film(context).chunk(2, -1)
            e = e * (1 + gamma[:, :, None]) + beta[:, :, None]
            hidden = states[:, -1:].transpose(0, 1).contiguous()
        dt = (chunk["time"][..., :, None] - chunk["keys_time"][..., None, :]).abs()
        parts = dict(
            shape=(m, c, k),
            y=e.reshape(m * c, k, -1),
            previous=pe.reshape(m * c, k, -1),
            bucket=torch.bucketize(dt, self.edges, right=True).view(m * c, k, 2 * k),
            keys_valid=chunk["keys_valid"].view(m * c, 2 * k),
            prior=((chunk["primary"].float() * 2 - 1) * math.log(9)).view(m * c, k),
        )
        for block in self.blocks[:-1]:
            parts["y"] = block(parts["y"], parts["previous"], parts["bucket"], parts["keys_valid"])
        return parts, hidden

    def readout(self, parts, sets=None, labels=None):
        """Last block and head, for all sets [m, C, K] or the selected ones [s, K].

        labels: (previous-batch states, batch states), each [sets, K], 0 masked / 1 / 2.
        """
        m, c, k = parts["shape"]
        names = ("y", "previous", "bucket", "keys_valid", "prior")
        y, previous, bucket, valid, prior = (
            parts[n] if sets is None else parts[n][sets] for n in names
        )
        terms = None
        if labels is not None:
            h = self.spec["heads"]
            embedded = self.label(torch.cat(labels, 1))
            terms = self.label_kv(embedded).view(len(y), 2 * k, 2, h, -1).permute(2, 0, 3, 1, 4)
        y = self.blocks[-1](y, previous, bucket, valid, terms)
        logits = prior + self.spec["bound"] * self.head(y).squeeze(-1).tanh()
        return logits.view(m, c, k) if sets is None else logits


def load_context(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = ContextLikelihood(**saved["spec"])
    model.load_state_dict(saved["model"])
    return model.eval()


def batch_posture(path):
    """Per-batch [q, q - q(2 s)] from the P1 posture store, checked against its manifest."""
    stored = P1 / "posture" / f"{key(path)}.npy"
    entry = read(P1 / "posture_manifest.json")["sources"][path.relative_to(OUTPUT).as_posix()]
    if sha256(stored) != entry["q_sha256"]:
        raise ValueError("Stored posture changed")
    q = np.load(stored).astype(np.float64)
    displacement = q - q[np.maximum(np.arange(len(q)) - LAG_BATCHES, 0)]
    return np.column_stack([q, displacement]).astype(np.float32)


class Layout:
    """Index matrix [units, batches, K] of a trajectory's candidates (-1 = padding)."""

    def __init__(self, events, batches):
        time, unit = np.asarray(events["time"]), np.asarray(events["unit"])
        batch = time // BATCH
        order = batch.astype(np.int64) * 4096 + unit
        if np.any(np.diff(order) < 0) or batch.max(initial=0) >= batches:
            raise ValueError("Candidates must be sorted by batch and unit inside the record")
        self.units = events["record"]["calibration"].spikes.shape[1]
        self.batches = batches
        counts = np.zeros((self.units, batches), dtype=np.int64)
        np.add.at(counts, (unit, batch), 1)
        self.index = np.full((self.units, batches, max(int(counts.max(initial=0)), 1)), -1)
        within = np.empty(len(time), dtype=np.int64)
        for j in range(self.units):
            idx = np.flatnonzero(unit == j)
            cells = batch[idx]
            within[idx] = np.arange(len(idx)) - np.searchsorted(cells, cells, side="left")
        self.index[unit, batch, within] = np.arange(len(time))


def chunks(layout, heads=4):
    """Consecutive batch ranges within CHUNK batches and BUDGET attention elements."""
    width = (layout.index >= 0).sum(2).max(0)
    a, ranges = 0, []
    while a < layout.batches:
        b = a + 1
        while b < min(a + CHUNK, layout.batches):
            k = int(width[max(a - 1, 0) : b + 1].max())
            if layout.units * (b + 1 - a) * k * 2 * k * heads > BUDGET:
                break
            b += 1
        ranges.append((a, b))
        a = b
    return ranges


def chunk_tensors(layout, values, a, b, device):
    """Batches [a, b); each batch also sees the previous batch's candidates as keys."""
    x24, offset, primary, posture = values
    width = max(int((layout.index[:, max(a - 1, 0) : b] >= 0).sum(2).max(initial=0)), 1)
    index = layout.index[:, a:b, :width]
    prev = np.full_like(index, -1)
    prev[:, 1:] = index[:, :-1]
    if a > 0:
        prev[:, 0] = layout.index[:, a - 1, :width]
    valid, pvalid = index >= 0, prev >= 0
    safe, psafe = index.clip(0), prev.clip(0)
    m = index.shape[0]
    time = np.rint(offset[safe] * BATCH).astype(np.int32)
    ptime = np.rint(offset[psafe] * BATCH).astype(np.int32) - BATCH
    start = (a + np.arange(b - a))[None, :, None] * BATCH
    arrays = dict(
        x=x24[safe] * valid[..., None],
        offset=offset[safe] * valid,
        primary=primary[safe] & valid,
        valid=valid,
        px=x24[psafe] * pvalid[..., None],
        poffset=offset[psafe] * pvalid,
        time=time,
        keys_time=np.concatenate([ptime, time], 2),
        keys_valid=np.concatenate([pvalid, valid], 2),
        posture=np.broadcast_to(posture[a:b], (m, b - a, posture.shape[1])),
        kept=valid & (start + time >= KEEP),
        pkept=pvalid & (start + ptime >= KEEP),
    )
    chunk = {n: torch.from_numpy(np.ascontiguousarray(v)).to(device) for n, v in arrays.items()}
    chunk["index"], chunk["valid_np"], chunk["prev_np"] = index, valid, prev
    return chunk


def trajectory_values(path, events):
    x24 = np.ascontiguousarray(inputs("posture24", path, events), dtype=np.float32)
    offset = ((np.asarray(events["time"]) % BATCH) / BATCH).astype(np.float32)
    primary = np.asarray(events["primary"], dtype=bool)
    return x24, offset, primary, batch_posture(path)


@torch.no_grad()
def predict(model, path, events, device="cpu", *, history=True, values=None):
    """Open-loop logits for every candidate, in the archive's order."""
    model = model.to(device).eval()
    if model.spec["kind"] == "wide":
        x24 = inputs("posture24", path, events) if values is None else values[0]
        parts = []
        for start in range(0, len(x24), 65536):
            x = torch.from_numpy(np.ascontiguousarray(x24[start : start + 65536])).to(device)
            p = torch.from_numpy(np.asarray(events["primary"][start : start + 65536])).to(device)
            parts.append(model({"x": x, "primary": p})[0].cpu().numpy())
        return np.concatenate(parts)
    values = trajectory_values(path, events) if values is None else values
    layout = Layout(events, len(values[3]))
    logits = np.zeros(len(values[0]), dtype=np.float32)
    hidden = carry = None
    for a, b in chunks(layout):
        chunk = chunk_tensors(layout, values, a, b, device)
        if model.spec["kind"] == "mdiff":
            out, hidden, carry = decode_labels(model, chunk, hidden, history, carry)
        else:
            out, hidden = model(chunk, hidden, history=history)
        valid = chunk["valid_np"]
        logits[chunk["index"][valid]] = out.cpu().numpy()[valid]
    return logits


def decode_labels(model, chunk, hidden, history=True, carry=None):
    """Batch by batch: the previous batch's decided labels are revealed; two passes reveal
    this batch by confidence; the last pass reads every candidate out given all others."""
    parts, hidden = model.trunk(chunk, hidden, history)
    m, c, k = parts["shape"]
    kept, device = chunk["kept"], chunk["kept"].device
    out = torch.zeros(m, c, k, device=device)
    previous = torch.zeros(m, k, dtype=torch.long, device=device)
    if carry is not None:
        width = min(k, carry.shape[1])
        previous[:, :width] = carry[:, :width]
    for t in range(c):
        sets = torch.arange(m, device=device) * c + t
        live = kept[:, t]
        state = torch.zeros(m, k, dtype=torch.long, device=device)
        remaining = live.clone()
        for step in range(STEPS):
            logits = model.readout(parts, sets, (previous, state))
            if step == STEPS - 1:
                break
            confidence = (logits.sigmoid() - 0.5).abs().masked_fill(~remaining, -1.0)
            left = remaining.sum(-1, keepdim=True).float()
            take = torch.ceil(left / (STEPS - 1 - step)).long()
            rank = confidence.argsort(-1, descending=True).argsort(-1)
            reveal = remaining & (rank < take)
            state = torch.where(reveal, 1 + (logits >= 0).long(), state)
            remaining = remaining & ~reveal
        out[:, t] = logits
        previous = torch.where(live, 1 + (logits >= 0).long(), 0)
    return out, hidden, previous


def unit_weights(events, keep):
    """memory_training.fit's unit balancing over post-calibration candidates."""
    unit = np.asarray(events["unit"])
    counts = np.bincount(unit[keep], minlength=events["record"]["calibration"].spikes.shape[1])
    weights = np.zeros(len(unit), dtype=np.float32)
    weights[keep] = len(unit[keep]) / (len(counts) * counts[unit[keep]].clip(1))
    return weights


def train_paths(pilot):
    training = l2_paths("training")
    if pilot:
        training = [(p, c) for p, c in training if "L2/standard/round3" in p.as_posix()]
    return training, l2_paths("validation")[-20:]


def validation_bce(model, validation_paths, validation, device):
    values = []
    for (path, _), events in zip(validation_paths, validation, strict=True):
        logits = predict(model, path, events, device)
        keep = events["keep"]
        loss = np.logaddexp(0, logits[keep]) - logits[keep] * events["target"][keep]
        m = events["record"]["calibration"].spikes.shape[1]
        values.extend(float(loss[events["unit"][keep] == j].mean()) for j in range(m))
    return float(np.mean(values))


def fit(arm, seed, *, folder=None, pilot=False, epochs=None):
    """P1's AdamW recipe; per-candidate arms use P1's i.i.d. loop, context arms chunks."""
    folder = folder or FOLDER / arm / f"seed{seed}"
    if epochs is None:
        epochs = EPOCHS if KINDS[arm] == "wide" else CONTEXT_EPOCHS
    if (folder / "complete.json").exists():
        if sha256(folder / "model.pt") != read(folder / "complete.json")["checkpoint_sha256"]:
            raise ValueError("Completed context classifier changed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized CUDA training environment")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    model = ContextLikelihood(KINDS[arm]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    training, validation_paths = train_paths(pilot)
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
    rng = np.random.default_rng(seed)
    history, best = [], float("inf")
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        losses = []
        model.train()
        for order in rng.permutation(len(training)):
            path, case = training[order]
            events = labelled(path, case, "training")
            keep = np.asarray(events["keep"])
            weights = unit_weights(events, keep)
            target = np.asarray(events["target"], dtype=np.float32)
            step = iid_epoch if model.spec["kind"] == "wide" else chunk_epoch
            losses += step(model, optimizer, path, events, keep, weights, target, rng)
        model.eval()
        value = validation_bce(model, validation_paths, validation, "cuda")
        history.append(dict(epoch=epoch, train_bce=float(np.mean(losses)), validation_bce=value))
        if value < best:
            best = value
            torch.save(
                dict(model=model.cpu().state_dict(), spec=model.spec, epoch=epoch),
                folder / "model.pt",
            )
            model.cuda()
        print("context C4", arm, seed, history[-1], now(), flush=True)
    write_new(folder / "history.json", history)
    write_new(
        folder / "complete.json",
        dict(
            completed_at_utc=now(),
            checkpoint_sha256=sha256(folder / "model.pt"),
            validation_bce=best,
            training_trajectories=len(training),
            epochs=epochs,
            seed=seed,
            parameters=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(),
        ),
    )


def iid_epoch(model, optimizer, path, events, keep, weights, target, rng):
    """memory_training.fit's loop for one trajectory, for per-candidate heads."""
    features = torch.from_numpy(inputs("posture24", path, events)[keep]).cuda()
    primary = torch.from_numpy(np.asarray(events["primary"])[keep]).cuda()
    goal = torch.from_numpy(target[keep]).cuda()
    w = torch.from_numpy(weights[keep]).cuda()
    losses = []
    permutation = torch.randperm(len(goal), device="cuda")
    for begin in range(0, len(goal), 16384):
        ids = permutation[begin : begin + 16384]
        x = features[ids] + torch.randn_like(features[ids]) * 0.02
        logits, _ = model({"x": x, "primary": primary[ids]})
        loss = F.binary_cross_entropy_with_logits(logits, goal[ids], reduction="none") * w[ids]
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def chunk_epoch(model, optimizer, path, events, keep, weights, target, rng):
    """Truncated BPTT over budget-bounded chunks of the trajectory, all units at once."""
    values = trajectory_values(path, events)
    layout = Layout(events, len(values[3]))
    losses, hidden = [], None
    for a, b in chunks(layout):
        chunk = chunk_tensors(layout, values, a, b, "cuda")
        for name, mask in (("x", "valid"), ("px", None)):
            noise = torch.randn_like(chunk[name]) * 0.02
            if mask is not None:
                noise = noise * chunk[mask][..., None]
            chunk[name] = chunk[name] + noise
        safe, valid = chunk["index"].clip(0), chunk["valid_np"]
        kept = torch.from_numpy(keep[safe] & valid).cuda()
        if not bool(kept.any()):
            if model.recurrent:
                with torch.no_grad():
                    _, hidden = model(chunk, hidden)
            continue
        goal = torch.from_numpy(target[safe]).cuda()
        w = torch.from_numpy(weights[safe]).cuda()
        labels = None
        if model.spec["kind"] == "mdiff":
            # Reveal each set's labels at its own U(0,1) rate, separately for the batch and
            # its previous batch; a candidate's own label never reaches its own logit.
            m, c, k = valid.shape
            prev_goal = torch.from_numpy(target[chunk["prev_np"].clip(0)]).cuda()
            terms = []
            for truth, mask in ((prev_goal, chunk["pkept"]), (goal, chunk["kept"])):
                rate = torch.from_numpy(rng.uniform(0, 1, (m, c, 1)).astype(np.float32)).cuda()
                reveal = (torch.rand(valid.shape, device="cuda") < rate) & mask
                terms.append(torch.where(reveal, 1 + truth.long(), 0).view(m * c, k))
            labels = tuple(terms)
        logits, hidden = model(chunk, hidden, labels=labels)
        bce = F.binary_cross_entropy_with_logits(logits, goal, reduction="none") * w
        loss = bce[kept].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if hidden is not None:
            hidden = hidden.detach()
        losses.append(float(loss.detach()))
    return losses


def select(arm):
    """posture_c4.select (memory_training.select) for a context head."""
    import optuna

    folder = FOLDER / arm
    if (folder / "selected.json").exists():
        return read(folder / "selected.json")
    validation_paths = l2_paths("validation")[-20:]
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_context(FOLDER / arm / f"seed{seed}/model.pt") for seed in SEEDS]
    logits = [
        [
            predict(model, path, events, device)
            for (path, _), events in zip(validation_paths, validation, strict=True)
        ]
        for model in models
    ]
    baseline_precision = read(OUTPUT / "L0/validation/summary.json")["precision_pct"]

    def measure(seed, threshold, strength):
        metrics = quick_scores(validation, logits[seed], threshold, strength)
        return objective_value(metrics, baseline_precision), metrics

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),
        storage=f"sqlite:///{(folder / 'tpe.sqlite3').as_posix()}",
        study_name="context_C4",
        load_if_exists=True,
    )

    def evaluate(trial):
        threshold = trial.suggest_float("threshold", 0.25, 0.85)
        strength = trial.suggest_float("strength", 0.0, 1.0)
        value, metrics = measure(0, threshold, strength)
        trial.set_user_attr("metrics", metrics)
        return value

    if not study.trials:
        study.enqueue_trial(dict(threshold=0.5, strength=0.0))
    study.optimize(evaluate, n_trials=max(0, 40 - len(study.trials)))
    top = sorted(
        (t for t in study.trials if t.value is not None), key=lambda t: t.value, reverse=True
    )[:3]
    repeats = []
    for trial in top:
        scores = [measure(seed, **trial.params) for seed in SEEDS]
        seed = int(np.argsort([value for value, _ in scores])[1])
        repeats.append(
            dict(
                trial=trial.number,
                parameters=trial.params,
                seed_metrics=[metrics for _, metrics in scores],
                median_objective=float(np.median([value for value, _ in scores])),
                selected_seed=seed,
            )
        )
    winner = max(repeats, key=lambda row: row["median_objective"])
    result = dict(
        selected_at_utc=now(),
        repeats=repeats,
        selection=winner,
        scope="L2 round-3 validation candidates, as in P1; final comparison uses other candidates",
    )
    write_new(folder / "selected.json", result)
    return result


def pilot(arms, epochs):
    """Unregistered pilot: L2 round-3 training rollouts only, seed 0, validation BCE/RoA."""
    root = OUTPUT / "P2_pilot"
    validation_paths = l2_paths("validation")[-20:]
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
    rows = {}
    for arm in arms:
        folder = root / arm
        fit(arm, 0, folder=folder, pilot=True, epochs=epochs)
        model = load_context(folder / "model.pt")
        logits = [
            predict(model, path, events, "cuda")
            for (path, _), events in zip(validation_paths, validation, strict=True)
        ]
        quick = {f"t{t}": quick_scores(validation, logits, t, 1.0) for t in (0.35, 0.45, 0.55)}
        rows[arm] = dict(complete=read(folder / "complete.json"), quick=quick)
        print(arm, rows[arm], flush=True)
    return rows


def variants():
    """Decision sets: every arm, plus what the recurrent heads lose without context."""
    sets = {arm: dict(arm=arm, history=True, frozen=False) for arm in ARMS}
    for arm in ("setrec24", "mdiff24"):
        sets[f"{arm}_nohistory"] = dict(arm=arm, history=False, frozen=False)
        sets[f"{arm}_frozenposture"] = dict(arm=arm, history=True, frozen=True)
    return sets


def frozen_values(values):
    """Posture held at the record median and displacement zero, in tokens and the GRU."""
    x24, offset, primary, posture = values
    x24 = x24.copy()
    x24[:, 22], x24[:, 23] = np.median(x24[:, 22]), 0.0
    posture = np.column_stack([np.full(len(posture), np.median(posture[:, 0])), 0 * posture[:, 1]])
    return x24, offset, primary, posture.astype(np.float32)


def decide():
    """Label-free open-loop decisions on the sealed L2 final candidates (P1's procedure)."""
    from .posture_c4 import final_events, hold_calibration
    from .train_likelihood import decode_record

    seal_path = FOLDER / "decisions/sealed.json"
    if seal_path.exists():
        return read(seal_path)
    reference = read(P1 / "decisions/sealed.json")
    for name, digest in reference["files"].items():
        if sha256(P1 / name) != digest:
            raise ValueError("Sealed P1 reference decision changed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    files, flips = {}, {}
    cases = [case for case in records(VALIDATION)]
    finals = [final_events(case) for case in cases]
    for name, spec in variants().items():
        parameters = read(FOLDER / spec["arm"] / "selected.json")["selection"]["parameters"]
        for seed in SEEDS:
            model = load_context(FOLDER / spec["arm"] / f"seed{seed}/model.pt")
            changed = total = 0
            for case, (path, events) in zip(cases, finals, strict=True):
                values = trajectory_values(path, events)
                if spec["frozen"]:
                    values = frozen_values(values)
                logits = predict(
                    model, path, events, device, history=spec["history"], values=values
                )
                binary = hold_calibration(
                    decode_record(events, logits, **parameters), events, BATCH, 2048
                )
                if name != spec["arm"]:
                    own = np.load(FOLDER / f"decisions/{spec['arm']}/seed{seed}/{stub(*case)}.npz")
                    changed += int(np.sum(own["spikes"][5 * 2048 :] != binary[5 * 2048 :]))
                    total += int(own["spikes"][5 * 2048 :].sum())
                destination = FOLDER / f"decisions/{name}/seed{seed}/{stub(*case)}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError("Orphaned decision: preserve it and use a fresh folder")
                np.savez_compressed(destination, spikes=binary)
                files[destination.relative_to(FOLDER).as_posix()] = sha256(destination)
            if name != spec["arm"]:
                flips[f"{name}_seed{seed}"] = dict(
                    changed_samples=changed, own_discharges=total, pct=changed / total * 100
                )
            print("decided", name, seed, now(), flush=True)
    result = dict(sealed_at_utc=now(), labels_opened=False, files=files, context_reliance=flips)
    write_new(seal_path, result)
    return result


def score_task(arguments):
    """Original evaluator plus P1's posture strata for one sealed decision file."""
    from .evaluate import score_record
    from .events import descriptor
    from .posture_c4 import posture_strata

    name, seed, case = arguments
    torch.set_num_threads(1)
    path = FOLDER / f"scores/{name}/seed{seed}/{stub(*case)}.json"
    if path.exists():
        return
    record = descriptor(*case, phase="validation")
    with np.load(FOLDER / f"decisions/{name}/seed{seed}/{stub(*case)}.npz") as archive:
        binary = archive["spikes"]
    result = score_record(record, dict(spikes=binary), phase="validation")
    strata = posture_strata(record, binary)
    for row in result["rows"]:
        row["strata"] = {k: v[row["unit"]] for k, v in strata.items()}
    write_new(path, result)


def contrasts():
    """Seed-mean contrasts (weights on '<set>_seed<s>' rows); P1 arms are references."""

    def mean(name, sign):
        return {f"{name}_seed{s}": sign / 3 for s in SEEDS}

    table = {f"{arm}_vs_posture24": {**mean(arm, 1), **mean("posture24", -1)} for arm in ARMS}
    table["posture24_vs_base22"] = {**mean("posture24", 1), **mean("base22", -1)}
    table["history_setrec24_vs_set24"] = {**mean("setrec24", 1), **mean("set24", -1)}
    table["generative_mdiff24_vs_setrec24"] = {**mean("mdiff24", 1), **mean("setrec24", -1)}
    for arm in ("setrec24", "mdiff24"):
        for cut in ("nohistory", "frozenposture"):
            table[f"{arm}_vs_{cut}"] = {**mean(arm, 1), **mean(f"{arm}_{cut}", -1)}
    return table


def score(workers=4):
    """Original evaluator on sealed decisions; paired cluster bootstrap, as in P1."""
    from collections import defaultdict
    from multiprocessing import Pool

    from .events import descriptor
    from .l2_ablation import combine, metrics

    path = FOLDER / "analysis.json"
    if path.exists():
        return read(path)
    seal = read(FOLDER / "decisions/sealed.json")
    for name, digest in seal["files"].items():
        if sha256(FOLDER / name) != digest:
            raise ValueError("Sealed open-loop decision changed")
    tasks = [(n, s, c) for n in variants() for s in SEEDS for c in records(VALIDATION)]
    with Pool(workers) as pool:
        pool.map(score_task, tasks, chunksize=1)
    rows, clusters = defaultdict(dict), defaultdict(list)
    for case in records(VALIDATION):
        calibration = descriptor(*case, phase="validation")["calibration"]
        for unit, index in enumerate(calibration.gt_matched_indices):
            clusters[(case[1], int(index))].append((case[1], case[2], unit))
    sources = {f"{n}_seed{s}": FOLDER / f"scores/{n}/seed{s}" for n in variants() for s in SEEDS}
    sources.update(
        {f"{a}_seed{s}": P1 / f"scores/{a}/seed{s}" for a in ("base22", "posture24") for s in SEEDS}
    )
    for label, folder in sources.items():
        for case in records(VALIDATION):
            for row in read(folder / f"{stub(*case)}.json")["rows"]:
                rows[label][(row["condition"], row["snr_db"], row["unit"])] = row
    keys, names = sorted(rows["posture24_seed0"]), sorted(clusters)
    table = contrasts()

    def summary(selected):
        values = {}
        for label, by_key in rows.items():
            chosen = [by_key[k] for k in selected]
            value = metrics(chosen)
            for stratum in ("near_reference", "far_20deg"):
                found = [r["strata"][stratum] for r in chosen if r["strata"][stratum] is not None]
                value[stratum] = float(np.mean(found) * 100) if found else None
            values[label] = value
        return values, {name: combine(w, values) for name, w in table.items()}

    point_values, point = summary(keys)
    rng = np.random.default_rng(2026)
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(names), len(names))
        _, contrast = summary([k for i in pick for k in clusters[names[i]]])
        for name, effect in contrast.items():
            for metric, value in effect.items():
                draws[name][metric].append(np.nan if value is None else value)
    effects = {
        name: {
            metric: dict(
                estimate=value,
                ci95=[float(v) for v in np.nanpercentile(draws[name][metric], [2.5, 97.5])],
            )
            for metric, value in effect.items()
        }
        for name, effect in point.items()
    }

    def above(name, metric="global_roa_pct"):
        return effects[name][metric]["ci95"][0] > 0

    bce = {
        arm: [read(FOLDER / arm / f"seed{s}/complete.json")["validation_bce"] for s in SEEDS]
        for arm in ARMS
    }
    bce["posture24"] = [
        read(P1 / f"posture24/seed{s}/complete.json")["validation_bce"] for s in SEEDS
    ]
    improving = [arm for arm in ARMS if above(f"{arm}_vs_posture24")]
    best = max(
        improving,
        key=lambda arm: effects[f"{arm}_vs_posture24"]["global_roa_pct"]["estimate"],
        default=None,
    )
    decisions = dict(
        improves_over_p1={arm: above(f"{arm}_vs_posture24") for arm in ARMS},
        capacity_alone_helps=above("wide24_vs_posture24"),
        history_helps=above("history_setrec24_vs_set24") and above("setrec24_vs_nohistory"),
        posture_used_by_setrec24=above("setrec24_vs_frozenposture"),
        generative_beats_discriminative=above("generative_mdiff24_vs_setrec24"),
        best_arm=best,
        propose_closed_loop_level=best is not None
        and all(b < p for b, p in zip(bce[best], bce["posture24"], strict=True)),
    )
    result = dict(
        analysed_at_utc=now(),
        instances=len(keys),
        clusters=len(names),
        arms=point_values,
        effects=effects,
        validation_bce=bce,
        selections={arm: read(FOLDER / arm / "selected.json")["selection"] for arm in ARMS},
        reliance=seal["context_reliance"],
        decisions=decisions,
    )
    write_new(path, result)
    return result


def register_protocol():
    if (FOLDER / "protocol.json").exists():
        return checked(FOLDER)
    register(
        FOLDER,
        dict(
            purpose="Open-loop test: do C4 heads that see the batch, the unit's history and "
            "the posture trajectory beat P1's per-candidate posture head; does a generative "
            "(masked discrete diffusion) decoder beat its discriminative twin?",
            arms={
                "wide24": "per-candidate MLP 24-64-64-1 (5,825 parameters), P1 i.i.d. loop",
                "set24": "per-unit attention over the batch's candidates plus the previous "
                "batch's as keys, 16-bucket time-separation bias, 2 blocks, width 32",
                "setrec24": "set24 + per-unit GRU across batches (mean/max of candidate "
                "embeddings, log count, q and 2 s displacement), FiLM on batch c from the "
                "state after batch c-1",
                "mdiff24": "setrec24 trunk; discharge labels of the batch and of the previous "
                "batch enter the last attention block as key/value terms, a candidate's own "
                "label never; trained by revealing each set's true labels at a U(0,1) rate "
                "(masked discrete diffusion / discrete flow matching with a mask source); "
                "decoded batch by batch: previous batch's decided labels (p >= 0.5), two "
                "confidence-ordered reveal passes, then the logit of every candidate given "
                "all other labels; calibration candidates are never revealed",
                "references": "P1 posture24 and base22 heads, selections and sealed decisions",
            },
            data="P1's: memory_training.paths(L2/standard, 3) training (420 trajectories, "
            "sub-01..03); epoch selection by validation BCE and TPE on L2 round-3 sub-04",
            inputs="P1's 24 (22 L2 + C1 q and q - q(2 s)); context heads also get the "
            "candidate's offset inside its batch; nothing else, no labels or angles",
            recipe=f"AdamW 1e-3, wd 0.01, unit-balanced BCE, feature noise 0.02, grad clip 1, "
            f"seeds 0/1/2, deterministic CUDA; wide24 {EPOCHS} epochs (P1); context heads "
            f"{CONTEXT_EPOCHS} epochs (compute: ~5 h; fewer epochs can only handicap them), "
            f"best epoch by validation BCE, <= {CHUNK}-batch "
            "truncated BPTT with the GRU state carried across chunks; TPE seed 2026, 40 "
            "trials, top-3 three-seed repeats, median seed (P1's select)",
            causality="batch c uses batches <= c only (tests: future perturbation, chunk "
            "length invariance, unit independence)",
            pilot="unregistered pilot on 60 L2 round-3 trajectories, seed 0, 3 epochs "
            "(outputs P2_pilot; quick validation RoA at threshold 0.35/0.45/0.55, strength "
            "1): P1-shape 24-8-1 head 90.04, wide24 90.96, set24 92.15, setrec24 92.28, "
            "first mdiff24 90.34 (reveal-time logits not comparable across reveal steps; "
            "replaced before registration by the own-label-free readout), redesigned "
            "mdiff24 91.49. Used to fix code, cost and decoding; no arm was dropped",
            evaluation="open-loop decisions on the sealed L2 final_validation candidates, "
            "calibration batches held at Pareto, sealed before the original evaluator",
            ablations="setrec24/mdiff24 with the GRU context zeroed (nohistory) and with "
            "posture frozen at the record median, displacement zero (frozenposture)",
            primary=f"seed-mean global RoA, arm - P1 posture24, paired cluster bootstrap "
            f"({BOOTSTRAP} draws, seed 2026, condition x ground-truth unit)",
            decisions=[
                "an arm improves C4 in open loop only if its primary 95% CI lies above zero",
                "history helps only if setrec24 beats set24 and its nohistory ablation (CIs)",
                "setrec24 uses posture only if it beats its frozenposture ablation (CI)",
                "generative training helps only if mdiff24 beats setrec24 (CI)",
                "propose a closed-loop level with the best improving arm only if its "
                "validation BCE is below posture24's for all three seeds",
                "nothing is deployed; no accepted level changes; sub-05 never opened",
            ],
            known_bias="sub-04 is the only validation subject and was already used for P1 "
            "and the post-hoc ceiling diagnostics (journal, 4 November)",
        ),
    )
    return checked(FOLDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["pilot", "register", "train", "select", "decide", "score"]
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=CONTEXT_EPOCHS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "pilot":
        print(pilot(args.arms, args.epochs))
    elif args.command == "register":
        print(sorted(register_protocol()["spec"]))
    elif args.command == "train":
        checked(FOLDER)
        for seed in args.seeds:
            fit(args.arm, seed)
    elif args.command == "select":
        checked(FOLDER)
        print(select(args.arm)["selection"])
    elif args.command == "decide":
        checked(FOLDER)
        print(decide()["context_reliance"])
    elif args.command == "score":
        checked(FOLDER)
        print(score(args.workers)["decisions"])
