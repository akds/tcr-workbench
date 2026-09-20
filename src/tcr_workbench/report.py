"""Atomic, auditable run bundles with bounded HTML previews."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from . import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot_inputs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Capture inputs before reading and detect changes while hashing."""
    result = {}
    for path in paths:
        path = Path(path).resolve()
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"Input changed while hashing: {path}")
        result[str(path)] = {"path": str(path), "sha256": digest, "size_bytes": after.st_size}
    return result


def verify_input_snapshot(snapshot: dict[str, dict[str, Any]]) -> None:
    for name, expected in snapshot.items():
        path = Path(name)
        if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Input changed during analysis: {path}; rerun from stable inputs")


def manifest(inputs: dict[str, Path], parameters: dict[str, Any], *,
             input_snapshot: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Hash inputs incrementally; a filename alone is never a cache identity."""
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"tcr_workbench": __version__, "python": platform.python_version(),
                     "source_sha256": source_digest(),
                     **{name: importlib.metadata.version(name)
                        for name in ("polars", "pyarrow", "pydantic", "rapidfuzz", "numpy")}},
        "inputs": {name: (input_snapshot or snapshot_inputs([path]))[str(Path(path).resolve())]
                   for name, path in inputs.items()},
        "parameters": parameters,
    }


@contextmanager
def output_bundle(destination: Path, *,
                  input_snapshot: dict[str, dict[str, Any]] | None = None) -> Iterator[Path]:
    """Publish only completed bundles; never clobber an existing analysis."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}. Choose a new --out directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        yield staging
        if input_snapshot:
            verify_input_snapshot(input_snapshot)
        if destination.exists():
            raise ValueError(f"Output appeared during analysis: {destination}")
        # Respect the process umask without changing that process-global setting.
        probe = staging / ".directory-permissions"
        probe.mkdir(mode=0o777)
        mode = stat.S_IMODE(probe.stat().st_mode)
        probe.rmdir()
        staging.chmod(mode)
        os.rename(staging, destination)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_frame(frame: pl.DataFrame, path: Path) -> None:
    """Parquet preserves types; CSV is the biologist-facing interchange file."""
    frame.write_parquet(path.with_suffix(".parquet"), compression="zstd")
    # Preserve nested data as JSON in tabular text instead of silently discarding it.
    nested = [name for name, dtype in frame.schema.items() if dtype.is_nested()]
    if nested:
        frame = frame.with_columns([
            pl.col(name).map_elements(
                lambda value: json.dumps(value.to_list() if hasattr(value, "to_list") else value),
                return_dtype=pl.String,
            ) for name in nested
        ])
    frame.write_csv(path)


def write_evidence_batches(batches: Iterator[pl.DataFrame], path: Path) -> tuple[pl.DataFrame, dict[str, int]]:
    """Stream sorted receptor blocks; retain only a 200-row HTML preview."""
    import pyarrow.parquet as pq

    preview = []
    preview_rows = evidence_rows = candidate_ids = 0
    previous_candidate = None
    writer = None
    schema = None
    try:
        with path.open("wb") as csv_stream:
            first = True
            for batch in batches:
                schema = batch.schema
                if writer is None:
                    writer = pq.ParquetWriter(path.with_suffix(".parquet"),
                                              batch.to_arrow().schema, compression="zstd")
                writer.write_table(batch.to_arrow())
                batch.write_csv(csv_stream, include_header=first)
                first = False
                evidence_rows += batch.height
                # The iterator guarantees receptor blocks are contiguous and ordered.
                ids = batch.filter(pl.col("status") == "Candidate")["receptor_id"].unique(maintain_order=True)
                for receptor_id in ids:
                    if receptor_id != previous_candidate:
                        candidate_ids += 1
                        previous_candidate = receptor_id
                if preview_rows < 200:
                    head = batch.head(200 - preview_rows)
                    # A zero-copy slice would pin all buffers of a huge receptor
                    # block. Copy only this bounded preview into independent buffers.
                    preview.append(pl.DataFrame(head.to_dict(as_series=False), schema=head.schema))
                    preview_rows += head.height
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Evidence iterator must yield its typed empty frame for empty inputs")
    return (pl.concat(preview, rechunk=False) if preview else pl.DataFrame(schema=schema),
            {"evidence_rows": evidence_rows, "receptors_with_candidates": candidate_ids})


_AA = "ACDEFGHIKLMNPQRSTVWY"
_SCORED_STATUSES = ("Scored", "ModelHypothesis")
_STYLE = """
:root{color-scheme:light;--navy:#102b3f;--teal:#087f78;--ink:#213e4d;--muted:#516875;--line:#d8e5e8}
*{box-sizing:border-box}body{margin:0;background:#f2f7f8;color:var(--ink);font:15px/1.6 system-ui,-apple-system,sans-serif}
a{color:#086d72;text-underline-offset:3px}a:hover{color:#10384c}a:focus-visible{outline:3px solid #edb653;outline-offset:4px}
.masthead{background:var(--navy);color:#fff;padding:25px max(24px,calc((100vw - 1240px)/2));border-bottom:4px solid #28afa1}
.brand{font-weight:750;font-size:20px;letter-spacing:-.5px}.brand span{color:#81e2d5}.brand small{font-size:11px;letter-spacing:1.5px;font-weight:500;display:block;color:#bfd6df;margin-top:3px}
main{max-width:1288px;margin:auto;padding:36px 24px 56px}.eyebrow{color:var(--teal);font-size:12px;letter-spacing:1.5px;font-weight:750;text-transform:uppercase;margin:0 0 8px}
h1{font-size:clamp(27px,4vw,40px);line-height:1.15;letter-spacing:-1.1px;color:var(--navy);margin:0 0 14px}h2{font-size:20px;color:var(--navy);margin:0 0 10px}h3{font-size:15px;margin:0 0 8px}
p{margin:0 0 14px}.lead{max-width:850px;font-size:17px;color:var(--muted)}.nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin:24px 0;font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:24px 0}.card,.panel{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 3px 12px #18364c05}
.card{padding:19px 22px;border-top:3px solid #35a99d}.value{font-size:28px;line-height:1.25;color:var(--navy);font-weight:720;overflow-wrap:anywhere}.label{font-size:12px;color:var(--muted);margin-top:5px;letter-spacing:.4px}
.card-warn{border-top-color:#bf8424;background:#fffdf8}.card-warn .value{color:#805219}
.panel{padding:26px;margin:20px 0}.note{background:#e9f5f2;border-left:4px solid var(--teal);padding:18px 22px;border-radius:0 10px 10px 0;margin:22px 0}.note p:last-child,.panel p:last-child{margin-bottom:0}
.nowrap{white-space:nowrap}.muted{color:var(--muted);font-size:13px}.scroll{overflow:auto;max-width:100%;border:1px solid var(--line);border-radius:9px}
table{border-collapse:collapse;width:100%;font-size:13px}caption{text-align:left;padding:12px;color:var(--muted)}th{background:#edf4f6;color:var(--navy);font-weight:650;white-space:nowrap}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}tr:last-child td,tr:last-child th{border-bottom:0}tbody tr:nth-child(even){background:#f8fbfc}td{max-width:400px;overflow-wrap:anywhere}
.status{display:inline-block;border-radius:100px;padding:3px 10px;font-size:11px;font-weight:700;white-space:nowrap;background:#edf1f4;color:#3c5260}.status-good{background:#dff4ec;color:#17624b}.status-open{background:#fff1da;color:#785019}
.metadata{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:18px 28px;margin:18px 0 0}.metadata div{min-width:0}.metadata dt{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:4px}.metadata dd{margin:0;font-size:14px;overflow-wrap:anywhere}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.files{display:flex;flex-wrap:wrap;gap:9px}.files a{display:inline-block;background:#f2f7f8;border:1px solid var(--line);padding:8px 12px;border-radius:7px;font-size:13px;text-decoration:none}.files a:hover{background:#e2f1ee}
.empty{padding:22px;background:#f7fafb;border:1px dashed #b4cbd2;border-radius:9px;color:var(--muted)}.heatmap{table-layout:fixed;min-width:1080px}.heatmap th,.heatmap td{text-align:center;padding:9px 3px;font-variant-numeric:tabular-nums}.heatmap th:first-child{width:72px}.heatmap td{font-size:11px;border:2px solid white}.prob-0{background:#eff8f7;color:#183d48}.prob-1{background:#d1eeea;color:#183d48}.prob-2{background:#91d5c9;color:#183d48}.prob-3{background:#40b5a4;color:#102b3f}.prob-4{background:#0f766e;color:white}
.axis{font:12px system-ui,-apple-system,sans-serif;fill:#516875}.motif-scroll{margin:16px 0}.motif-scroll svg{display:block;width:100%;max-height:350px}.consensus{overflow-wrap:anywhere;font-size:17px;letter-spacing:2px}.walkthrough{margin-top:24px;padding-top:20px;border-top:1px solid #c4dfd8}.walkthrough ol{padding-left:24px}.walkthrough li{padding-left:5px;margin:0 0 16px}.column-guide{margin-bottom:0}.column-guide div{display:grid;grid-template-columns:145px minmax(0,1fr);gap:20px;padding:12px 0;border-bottom:1px solid var(--line)}.column-guide div:last-child{border-bottom:0}.column-guide dd{margin:0}.comparison{padding:22px 0;border-top:1px solid var(--line)}.comparison svg{display:block;width:100%}.comparison h3{overflow-wrap:anywhere}.comparison:first-of-type{border-top:0;padding-top:10px}.distribution-title{margin-top:26px}.unresolved{margin-top:24px;background:#fff7e9;padding:16px;border-radius:9px}.unresolved summary{cursor:pointer;font-weight:650;margin-bottom:12px}.legend{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0}.legend span{padding:4px 9px;border-radius:4px;font-size:11px}.footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:18px;margin-top:28px}
@media(max-width:640px){main{padding:24px 16px}.panel{padding:18px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.value{font-size:24px}.metadata{grid-template-columns:1fr}.column-guide div{grid-template-columns:1fr;gap:4px}th,td{padding:9px}}
@media print{body{background:white}.masthead{background:white;color:#102b3f;padding:10px 0}.brand span,.brand small{color:#102b3f}main{padding:12px 0}.nav{display:none}.panel,.card{box-shadow:none;break-inside:avoid}.scroll{overflow:visible}.heatmap{min-width:0}.heatmap td{font-size:7px}.files a{padding:3px}.footer{margin-top:12px}}
"""


def _text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Unavailable (non-finite)"
        return format(value, ".6g")
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _table(frame: pl.DataFrame, columns: list[str] | None = None) -> str:
    selected = [name for name in (columns or frame.columns) if name in frame.columns]
    if not frame.height or not selected:
        return '<div class="empty">No result rows are available. Read the status and reason above; an empty result is not evidence of non-binding.</div>'
    heading = "".join(f'<th scope="col">{_escape(name.replace("_", " "))}</th>' for name in selected)
    rows = []
    for row in frame.select(selected).head(200).iter_rows():
        cells = []
        for name, value in zip(selected, row):
            text = _escape(value)
            if name == "status":
                style = "status-good" if value in {*_SCORED_STATUSES, "Profiled", "Candidate"} else "status-open"
                text = f'<span class="status {style}">{text}</span>'
            attribute = ' class="nowrap"' if name in {"peptide", "hla", "score", "rank", "peptide_length"} else ""
            cells.append(f"<td{attribute}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll" tabindex="0" role="region" aria-label="Result table"><table><thead><tr>{heading}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _profile_heatmap(profile: pl.DataFrame) -> str:
    if not profile.height:
        return '<div class="empty">No amino-acid profile was produced for this context. The unresolved reason is recorded above and in the output files.</div>'
    if not {"position", *_AA} <= set(profile.columns):
        return '<div class="empty">The profile cannot be visualized because required amino-acid columns are missing. Inspect the full profile file.</div>'
    rows = []
    for row in profile.select("position", *_AA).head(200).iter_rows():
        cells = [f'<th scope="row">{_escape(row[0])}</th>']
        for aa, value in zip(_AA, row[1:]):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                return '<div class="empty">The heatmap is unavailable because the profile contains missing or invalid probabilities. Inspect the full profile file.</div>'
            band = sum(value >= threshold for threshold in (0.05, 0.10, 0.25, 0.50))
            label = _escape(f"Position {_text(row[0])}, {aa}: {value:.6g}")
            cells.append(f'<td class="prob-{band}" title="{label}" aria-label="{label}">{value:.1%}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    heading = '<th scope="col">Position</th>' + "".join(f'<th scope="col">{aa}</th>' for aa in _AA)
    return ('<div class="scroll" tabindex="0" role="region" aria-label="Amino-acid probability heatmap">'
            f'<table class="heatmap"><caption>Conditional probabilities over 20 canonical amino acids; positions are one-based.</caption><thead><tr>{heading}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '<div class="legend" aria-label="Probability color scale">' + "".join(
                f'<span class="prob-{i}">{label}</span>' for i, label in enumerate(
                    ("0–&lt;5%", "5–&lt;10%", "10–&lt;25%", "25–&lt;50%", "50–100%"))) + '</div>')


# Geometric letter outlines keep the logo entirely self-contained and independent
# of installed fonts. Coordinates and strokes have the same vertical extent.
_LETTER_PATHS = {
    "A": "M.1,.9 L.5,.1 L.9,.9 M.26,.58 L.74,.58",
    "C": "M.87,.22 Q.5,-.03 .16,.26 Q0,.5 .16,.74 Q.5,1.03 .87,.78",
    "D": "M.16,.1 L.16,.9 L.48,.9 Q.92,.9 .92,.5 Q.92,.1 .48,.1 Z",
    "E": "M.88,.1 L.15,.1 L.15,.9 L.88,.9 M.15,.5 L.76,.5",
    "F": "M.88,.1 L.15,.1 L.15,.9 M.15,.5 L.76,.5",
    "G": "M.87,.22 Q.5,-.03 .16,.26 Q0,.5 .16,.74 Q.5,1.03 .88,.78 L.88,.52 L.56,.52",
    "H": "M.15,.1 L.15,.9 M.85,.1 L.85,.9 M.15,.5 L.85,.5",
    "I": "M.15,.1 L.85,.1 M.5,.1 L.5,.9 M.15,.9 L.85,.9",
    "K": "M.15,.1 L.15,.9 M.86,.1 L.15,.56 M.43,.4 L.88,.9",
    "L": "M.15,.1 L.15,.9 L.88,.9",
    "M": "M.1,.9 L.1,.1 L.5,.55 L.9,.1 L.9,.9",
    "N": "M.13,.9 L.13,.1 L.87,.9 L.87,.1",
    "P": "M.15,.9 L.15,.1 L.6,.1 Q.91,.1 .91,.34 Q.91,.57 .6,.57 L.15,.57",
    "Q": "M.5,.1 Q.12,.1 .12,.5 Q.12,.9 .5,.9 Q.88,.9 .88,.5 Q.88,.1 .5,.1 Z M.58,.65 L.9,.95",
    "R": "M.15,.9 L.15,.1 L.6,.1 Q.91,.1 .91,.33 Q.91,.55 .6,.55 L.15,.55 M.52,.55 L.9,.9",
    "S": "M.87,.2 Q.62,-.01 .25,.15 Q-.04,.36 .5,.5 Q1.04,.64 .75,.85 Q.38,1.01 .13,.8",
    "T": "M.07,.1 L.93,.1 M.5,.1 L.5,.9",
    "V": "M.08,.1 L.5,.9 L.92,.1",
    "W": "M.07,.1 L.25,.9 L.5,.45 L.75,.9 L.93,.1",
    "Y": "M.08,.1 L.5,.5 L.92,.1 M.5,.5 L.5,.9",
}
_AA_COLORS = {aa: color for letters, color in (
    ("AVLIMFWY", "#243e69"), ("STNQ", "#087f78"), ("KRH", "#8052aa"),
    ("DE", "#bf4955"), ("CGP", "#b47816")) for aa in letters}


def _profile_logo(profile: pl.DataFrame) -> str:
    """AA20 information logo; no calibrated binding or signed-PSSM claim."""
    if not profile.height:
        return '<div class="empty">No motif is available because this context was unresolved. An empty motif does not mean that no peptide binds.</div>'
    if not {"position", *_AA} <= set(profile.columns):
        return '<div class="empty">The motif requires all 20 amino-acid probability columns. Inspect the profile file.</div>'
    rows = list(profile.select("position", *_AA).head(200).iter_rows())
    for row in rows:
        values = row[1:]
        if (any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in values)
                or not math.isclose(sum(values), 1.0, abs_tol=1e-5)):
            return '<div class="empty">The motif is unavailable because the profile contains invalid probabilities or positions that do not sum to one.</div>'
    maximum = math.log2(20)
    consensus = []
    charts = []
    for offset in range(0, len(rows), 25):
        subset = rows[offset:offset + 25]
        width, height, left, bottom, scale, step = max(620, 70 + len(subset) * 43), 310, 52, 244, 46, 43
        marks = []
        for tick in range(5):
            y = bottom - tick * scale
            marks.append(f'<path d="M{left},{y} H{width-16}" stroke="#e3ebee"/><text x="{left-10}" y="{y+4}" text-anchor="end" class="axis">{tick}</text>')
        marks.append('<text x="15" y="145" transform="rotate(-90 15 145)" class="axis">Information (bits)</text>')
        for index, row in enumerate(subset):
            position, values = row[0], row[1:]
            information = max(0.0, maximum + sum(p * math.log2(p) for p in values if p))
            best = max(values)
            favored = [aa for aa, p in zip(_AA, values) if math.isclose(p, best, abs_tol=1e-12)]
            consensus.append(favored[0] if len(favored) == 1 else "X")
            x, baseline = left + index * step + 6, bottom
            for aa, probability in sorted(zip(_AA, values), key=lambda item: (item[1], item[0])):
                letter_height = probability * information * scale
                if letter_height <= 0.005:
                    continue
                baseline -= letter_height
                label = f"Position {_text(position)}; {aa}: {probability:.2%}; {probability * information:.4g} bits"
                marks.append(f'<g aria-label="{_escape(label)}"><title>{_escape(label)}</title><path d="{_LETTER_PATHS[aa]}" transform="translate({x:.3f} {baseline:.3f}) scale(32 {letter_height:.5f})" fill="none" stroke="{_AA_COLORS[aa]}" stroke-width=".16" stroke-linecap="butt" stroke-linejoin="round"/></g>')
            marks.append(f'<text x="{x+16}" y="{bottom+22}" text-anchor="middle" class="axis">{_escape(position)}</text>')
        marks.append(f'<text x="{width/2}" y="{height-13}" text-anchor="middle" class="axis">Peptide position, starting at 1 (N terminus → C terminus)</text>')
        charts.append(f'<div class="scroll motif-scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="min-width:{min(width,900)}px" role="img" aria-label="Information-content amino-acid sequence logo"><title>Model-derived amino-acid sequence logo</title>{"".join(marks)}</svg></div>')
    limit = f'<p class="muted">Showing the first 200 of {profile.height:,} positions; the full profile is in the download.</p>' if profile.height > 200 else ''
    return ('<p class="muted">On a narrow screen, scroll the motif sideways to see every position.</p>' + ''.join(charts) + limit
            + '<div class="legend residue-legend"><span style="color:#243e69">AVLIMFWY · hydrophobic/aromatic</span><span style="color:#087f78">STNQ · polar</span><span style="color:#8052aa">KRH · basic</span><span style="color:#bf4955">DE · acidic</span><span style="color:#b47816">CGP · other</span></div>'
            + '<p><strong>Highest-preference residue at each position:</strong> <code class="consensus">' + ''.join(consensus)
            + '</code>. X means the highest preference is tied. This sequence is a position-by-position summary, not a tested or optimized binder.</p>'
            + '<p class="muted">Logo definition: stack height = log₂(20) − H, where H = −Σ p log₂(p); each letter height = p × stack height. The maximum is 4.322 bits. A blank or tiny stack means broad amino-acid preference, not missing data. These are model probabilities, so no sample-size correction is applied. Letter colors group amino-acid properties only.</p>'
            + '<p class="muted">This information-content logo visualizes the probability profile used to make the PSSM. Letter heights are not signed PSSM scores. The downloadable PSSM gives log₂ odds for each amino acid against the uniform 5% background.</p>')


def _workflow_guide(workflow: str | None, background_metadata: dict | None = None) -> str:
    if workflow in {"pmhc-profile", "tcr-profile"}:
        target = "the selected MHC molecule" if workflow == "pmhc-profile" else "the paired TCR and selected MHC molecule"
        heading = "Profile definitions"
        steps = [
            ("Conditioning context", f"The profile gives amino-acid preferences conditioned on {target} and the specified peptide length. The species, input sequences and checkpoint determine the context; checkpoint training coverage should match the intended analysis."),
            ("Positions and letter heights", "Position 1 is the peptide N terminus. Within each stack, larger letters indicate higher model probability. Taller stacks indicate more concentrated preferences; a short stack does not establish that a position is biologically irrelevant."),
            ("Amino-acid probabilities", "The heatmap gives probabilities normalized over the 20 canonical amino acids at each position. A value of 50% refers to that amino-acid distribution, not a 50% probability of peptide binding."),
            ("Independent-position assumption", "DecoderTCR computes the profile with the entire peptide masked simultaneously. Combining favored residues does not model interactions between peptide residues. The corresponding score workflow averages context-conditioned log probabilities and does not restore those interactions."),
            ("Biological scope", "The profile is a model prediction, not an experimentally measured binding motif. No class II binding-core register is inferred. Affinity, antigen processing, cell-surface presentation and T-cell activation require separate experimental evidence."),
        ]
    elif workflow in {"pmhc-score", "tcr-score", "repertoire-score"}:
        target = "MHC molecule" if workflow == "pmhc-score" else "receptor and MHC molecule"
        heading = "Score definitions"
        steps = [
            ("Comparison groups", f"Scores and ranks are comparable only within the same species, {target}, peptide length, checkpoint and precision. Each group below is a separate comparison."),
            ("Score and rank", "Higher PLL, usually less negative, indicates greater model compatibility with the specified context. Rank 1 is the highest score among successfully scored tested peptides. Neither score nor rank is a probability of binding, a measured affinity or a binary binding classification."),
            ("Random-peptide reference", "Where available, the distribution compares tested peptides with random sequences scored by the same model in the same context and at the same length. Higher PLL indicates greater model preference relative to this synthetic reference. Random sequences are not experimentally confirmed nonbinders; the comparison provides no validated binding threshold or statistical significance."),
            ("Status and reason", "Scored or ModelHypothesis denotes a finite model score. Unresolved means that scoring or supporting evidence was unavailable, not that the peptide is nonbinding. The reason column identifies relevant limitations; complete tables retain unresolved inputs."),
            ("Biological scope", "MHC binding, presentation and T-cell recognition are distinct biological outcomes. Model scores alone do not establish any of them; interpretation requires appropriate experimental assays and controls."),
        ]
        if workflow == "repertoire-score":
            steps[2] = ("Receptor and cell mapping", "The receptor_id links each row to its reconstructed chains and source cells. Alternative chain pairings represent receptor hypotheses, not independent cells; complete tables retain pairings and unresolved inputs.")
        elif (background_metadata or {}).get("mode") == "mhc-profile":
            steps[2] = ("MHC-profile reference", "Reference peptides are sampled independently at each position from the MHC-only profile, with replacement at temperature 1. No TCR is used to generate them. They are then scored in the same MHC or TCR–MHC context as the tested peptides. This comparison measures model preference relative to MHC-profile samples; it does not establish binding or receptor specificity.")
    else:
        return ""
    return (f'<div class="walkthrough"><h3>{heading}</h3><ol>' + ''.join(
        f'<li><strong>{heading}.</strong> {_escape(body)}</li>' for heading, body in steps) + '</ol></div>')


def _column_guide(columns: list[str]) -> str:
    descriptions = {
        "receptor_id": "Identifier linking a receptor or alternative pairing to the receptor and cell tables. Alternative chain pairings can share source cells.",
        "hla": "MHC molecule used for the analysis. Human molecules use HLA names; supported mouse molecules use H-2 names. The field remains hla for compatibility. It specifies the analysis context, not necessarily the donor's complete MHC genotype.",
        "peptide": "Peptide amino-acid sequence, written in one-letter code from N terminus to C terminus.",
        "peptide_length": "The number of amino acids in this peptide. Compare PLL scores and ranks within one peptide length.",
        "score": "DecoderTCR peptide pseudo-log-likelihood (PLL): mask all peptide residues simultaneously, predict residues from the MHC or TCR–MHC context, and average the natural log of the full-vocabulary probabilities assigned to the tested peptide residues. This is not leave-one-residue-out PLL and does not model interactions between peptide residues. Higher (often less negative) indicates greater model compatibility within the same context. It is not an affinity or binding probability; there is no validated universal cutoff.",
        "rank": "Position among successfully scored tested peptides in the same receptor, MHC and peptide-length group. Rank 1 is highest; ties share a rank. The HTML score preview ranks distinct peptides. A rank is not a percentile or a probability.",
        "background_percentile": "Upper-tail reference percentile: 100 × (1 + number of reference scores greater than or equal to this score) / (N + 1). Lower values indicate a higher score relative to the stated reference distribution. This is descriptive, not a binding probability or calibrated significance test. PLL and panel rank remain the main outputs.",
        "background_n": "Number of finite scored reference draws in this comparison group, including repeats. With 1,000 draws, the smallest reported upper-tail percentile is approximately 0.1%; extreme tails have limited sampling precision.",
        "status": "Scored or ModelHypothesis: a finite model score was returned. Profiled: a probability profile was produced. Candidate: reference evidence supports a hypothesis. Unresolved: the available input or evidence was insufficient. None is a definitive binding label.",
        "reason": "Explanation of unresolved inputs, exclusions or reconstruction assumptions. A blank reason does not provide additional biological evidence.",
    }
    selected = [key for key in descriptions if key in columns]
    if not selected:
        return ""
    return ('<section class="panel" id="columns"><h2>Column definitions</h2><dl class="column-guide">'
            + ''.join(f'<div><dt><code>{key}</code></dt><dd>{_escape(descriptions[key])}</dd></div>' for key in selected) + '</dl></section>')


def _score_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if not {"peptide", "score"} <= set(frame.columns):
        return pl.DataFrame()
    frame = frame.with_columns(pl.col("score").cast(pl.Float64, strict=False),
                              pl.col("peptide").cast(pl.String).str.len_chars().alias("peptide_length"))
    condition = pl.col("score").is_finite() & pl.col("peptide").is_not_null()
    if "status" in frame.columns:
        condition = condition & pl.col("status").is_in(_SCORED_STATUSES)
    return frame.filter(condition)


def _histogram(tested: pl.DataFrame, background: pl.DataFrame | None, *, excluded: int = 0,
               reference_label: str = "Random reference") -> str:
    row_label = "random-reference" if reference_label == "Random reference" else "MHC-profile reference"
    unavailable_label = "Random-peptide reference" if reference_label == "Random reference" else reference_label
    coverage = (f'<p class="muted">{excluded:,} {_escape(row_label)} rows in this group were unresolved or had no finite score and are excluded from the distribution. Inspect background_scores.csv for the reasons.</p>' if excluded else "")
    if background is None or not background.height:
        return coverage + f'<div class="empty">{_escape(unavailable_label)} unavailable: no finite, same-context background scores were produced or supplied. The distribution is omitted.</div>'
    import numpy as np

    observed, reference = tested["score"].to_numpy(), background["score"].to_numpy()
    lower, upper = min(observed.min(), reference.min()), max(observed.max(), reference.max())
    if lower == upper:
        padding = max(abs(float(lower)) * 0.02, 0.1)
        lower, upper = lower - padding, upper + padding
    bins = np.linspace(lower, upper, 25)
    test_count, _ = np.histogram(observed, bins=bins)
    ref_count, _ = np.histogram(reference, bins=bins)
    test_fraction, ref_fraction = test_count / len(observed), ref_count / len(reference)
    peak = max(float(test_fraction.max()), float(ref_fraction.max()), 0.01)
    width, left, top, height, plot_width = 740, 62, 26, 190, 644
    marks = []
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = top + height * (1 - fraction)
        marks.append(f'<path d="M{left},{y:.3f} H{left+plot_width}" stroke="#dfe8ec"/><text x="{left-9}" y="{y+4:.3f}" text-anchor="end" class="axis">{fraction*peak:.0%}</text>')
    for index, (test, ref) in enumerate(zip(test_fraction, ref_fraction)):
        x, bar_width = left + plot_width * index / 24, plot_width / 24
        for offset, value, color, label in ((0, ref, "#aab9c7", reference_label), (bar_width / 2, test, "#087f78", "Tested peptides")):
            bar_height = height * value / peak
            tooltip = f"{label}: {value:.2%}; PLL {bins[index]:.4g} to {bins[index+1]:.4g}"
            marks.append(f'<rect x="{x+offset:.3f}" y="{top+height-bar_height:.3f}" width="{bar_width/2-1:.3f}" height="{bar_height:.3f}" fill="{color}"><title>{_escape(tooltip)}</title></rect>')
    for index in (0, 6, 12, 18, 24):
        x = left + plot_width * index / 24
        marks.append(f'<text x="{x:.3f}" y="{top+height+24}" text-anchor="middle" class="axis">{bins[index]:.3g}</text>')
    marks.append('<text x="386" y="276" text-anchor="middle" class="axis">Peptide PLL (natural-log units) → higher model compatibility</text>')
    marks.append('<text x="16" y="123" transform="rotate(-90 16 123)" text-anchor="middle" class="axis">Fraction of each set</text>')
    return (coverage + f'<div class="scroll"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 302" style="min-width:600px" role="img" aria-label="Tested versus reference-peptide PLL distribution"><title>PLL distribution: same context and peptide length</title>{"".join(marks)}</svg></div>'
            + f'<div class="legend"><span style="background:#e1f3ed;color:#08645f">Tested peptides · n = {len(observed):,}</span><span style="background:#e9eef3;color:#3e5468">{_escape(reference_label)} · n = {len(reference):,}</span></div>'
            + '<p class="muted">Each set is normalized separately to 100%; bars use identical score bins. Tested duplicates count once per peptide in this group. Background rows are random draws and may include repeats. Higher PLL is to the right. A synthetic reference is not a set of known nonbinders.</p>')


def _score_preview(table: pl.DataFrame, columns: list[str] | None,
                   background: pl.DataFrame | None,
                   background_metadata: dict | None = None) -> tuple[str, list[str]]:
    scored = _score_frame(table)
    background_input = (background.with_columns(pl.col("peptide").cast(pl.String).str.len_chars().alias("peptide_length"))
                        if background is not None and "peptide" in background.columns else None)
    background = _score_frame(background) if background is not None else None
    if not scored.height:
        return ('<div class="empty">No finite scored peptide is available for ranking or a PLL distribution. Review unresolved rows in the complete results.</div>'
                + _table(table.head(10), columns), columns or table.columns)
    groups = [key for key in ("receptor_id", "hla", "peptide_length") if key in scored.columns]
    unique = scored.unique(subset=[*groups, "peptide"], maintain_order=True)
    contexts = unique.select(groups).unique(maintain_order=True)
    selected_columns = [key for key in ("receptor_id", "hla", "peptide", "peptide_length", "score", "rank", "background_percentile", "background_n", "status", "reason")
                        if key in unique.columns or key == "rank"]
    panels = []
    profile_reference = (background_metadata or {}).get("mode") == "mhc-profile"
    reference_label = "MHC-profile reference" if profile_reference else "Random reference"
    distribution_title = "Tested peptides versus " + ("MHC-profile reference" if profile_reference else "random-peptide reference")
    if (background_metadata or {}).get("failed_contexts"):
        panels.append('<p class="muted">Some MHC profiles could not be generated. No alternative distribution was substituted; inspect background_metadata.json for the affected contexts and reasons.</p>')
    for context in contexts.head(20).iter_rows(named=True):
        predicate = pl.all_horizontal([pl.col(key).eq_missing(value) for key, value in context.items()])
        peptides = unique.filter(predicate).with_columns(
            pl.col("score").rank(method="min", descending=True).cast(pl.UInt32).alias("rank"))
        top = peptides.sort(["score", "peptide"], descending=[True, False]).head(10)
        label = " · ".join(f'{key.replace("_", " ")}: {_text(value)}' for key, value in context.items())
        matching, excluded = None, 0
        if background_input is not None and set(groups) <= set(background_input.columns):
            excluded = background_input.filter(predicate).height
        if background is not None and background.height and set(groups) <= set(background.columns):
            matching = background.filter(predicate)
            excluded -= matching.height
        panels.append(f'<div class="comparison"><h3>{_escape(label)}</h3><p class="muted">Top {top.height} of {peptides.height:,} distinct successfully scored tested peptides. Higher PLL comes first; ties share a rank and are displayed alphabetically.</p>'
                      + _table(top, selected_columns) + f'<h3 class="distribution-title">{distribution_title}</h3>'
                      + _histogram(peptides, matching, excluded=excluded, reference_label=reference_label) + '</div>')
    if contexts.height > 20:
        panels.append(f'<p class="muted">Showing 20 of {contexts.height:,} comparison groups to keep the report small. All groups remain in the complete tables.</p>')
    if "status" in table.columns:
        unresolved = table.filter(~pl.col("status").is_in(_SCORED_STATUSES).fill_null(False))
        if unresolved.height:
            panels.append(f'<details class="unresolved"><summary>{unresolved.height:,} input rows were not scored — show reasons</summary>'
                          + _table(unresolved.head(10), [key for key in selected_columns if key != "rank"])
                          + '<p class="muted">Showing at most 10 unscored rows. The complete results retain every input and reason.</p></details>')
    return ''.join(panels), selected_columns


def profile_reconstruction_context(components: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    """Expose reported reconstruction choices; never invent a selected allele."""
    from .prediction import _annotate_gene_resolution

    genes = ("trav", "traj", "trbv", "trbj")
    reconstruction = execution.get("reconstruction") or {}
    values = {gene: components.get(gene) for gene in genes}
    values.update({gene.upper(): reconstruction.get(gene.upper()) for gene in genes})
    values.update(status="ModelHypothesis", reason="")
    frame = pl.DataFrame([values], schema={key: pl.String for key in values})
    audit = _annotate_gene_resolution(frame).row(0, named=True)
    context = {f"Reconstructed {gene.upper()} (reported)": reconstruction.get(gene.upper()) or "Not reported"
               for gene in genes}
    context["Gene reconstruction check"] = (audit["gene_resolution_reason"] or
                                            "Reported V/J genes match the supplied calls.")
    warnings = [str(reconstruction[key]) for key in ("tcr_reason", "hla_reason") if reconstruction.get(key)]
    if warnings:
        context["Reconstruction messages"] = "; ".join(warnings)
    return context


def model_report_metadata(options: dict[str, Any], execution: dict[str, Any] | None = None) -> dict[str, Any]:
    """Show the selected runtime without exposing interpreter or checkpoint paths."""
    execution = execution or {}
    fields = {"Model": execution.get("model") or options.get("model"),
              "Device": execution.get("device") or options.get("device"),
              "Precision": execution.get("precision") or options.get("precision", "float32")}
    if fields["Precision"] == "float16":
        fields["Precision"] = "float16 (approximate)"
    for key in ("checkpoint_sha256", "source_checkpoint_sha256", "bundle_sha256"):
        if execution.get(key):
            fields["Checkpoint identity"] = execution[key]
            break
    fields["Execution"] = execution.get("status", "Completed; inspect row-level status")
    return fields


def write_workflow_report(directory: Path, title: str, *, table: pl.DataFrame | None = None,
                          columns: list[str] | None = None, summary: dict[str, Any] | None = None,
                          metadata: dict[str, Any] | None = None, interpretation: str,
                          profile: pl.DataFrame | None = None, context: dict[str, Any] | None = None,
                          total_rows: int | None = None, reason: str = "", manifest_pending: bool = False,
                          workflow: str | None = None, background: pl.DataFrame | None = None,
                          background_metadata: dict | None = None,
                          embedding_table: pl.DataFrame | None = None,
                          embedding_total_rows: int | None = None) -> None:
    """Render an offline, escaped, bounded report; callers retain manifest ownership.

    Publication callers set manifest_pending only when they write the manifest
    immediately afterward, including this report's hash in its output inventory.
    """
    directory = Path(directory)
    cards = []
    for key, value in (summary or {}).items():
        style = " card-warn" if "unresolved" in key.lower() or value == "Unresolved" else ""
        formatted = f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else value
        cards.append(f'<div class="card{style}"><div class="value">{_escape(formatted)}</div><div class="label">{_escape(key.replace("_", " "))}</div></div>')
    cards = "".join(cards)
    details = "".join(f'<div><dt>{_escape(key)}</dt><dd>{_escape(value)}</dd></div>'
                      for key, value in {**(metadata or {}), **(context or {})}.items())
    preferred = ("results.csv", "results.parquet", "profile.csv", "profile.parquet", "pssm.csv", "pssm.parquet",
                 "evidence.csv", "evidence.parquet", "receptors.csv", "cells.csv", "chains.csv", "qc.json",
                 "summary.json", "pssm_metadata.json", "background_scores.csv", "background_scores.parquet",
                 "background_metadata.json", "embedding_matches.csv", "embedding_matches.parquet",
                 "embedding_audit.csv", "embedding_metadata.json", "embedding_model_manifest.json",
                 "embedding_reconstruction_audit.csv", "decoder_input.csv.skipped.csv", "manifest.json")
    links = "".join(f'<a href="{name}">{name}</a>' for name in preferred
                    if (directory / name).is_file() or (name == "manifest.json" and manifest_pending))
    reason_html = f'<p><strong>Reason:</strong> {_escape(reason)}</p>' if reason else ""
    chart = ""
    if profile is not None:
        chart = ('<section class="panel" id="profile"><h2>Amino-acid probability heatmap</h2>'
                 '<p class="muted">Darker cells indicate greater model preference within a position. Values are not measured binding probabilities. No class II binding-core register is inferred.</p>'
                 + _profile_heatmap(profile) + '<p class="muted">PSSM legend: log₂(max(probability, 10⁻¹²) / 0.05). Zero matches the uniform amino-acid background; positive values indicate enrichment and negative values indicate depletion. The floor applies only to log-odds. This AA20 profile cannot recover the absolute full-vocabulary PLL.</p></section>')
    preview = ""
    displayed_columns = columns or (table.columns if table is not None else [])
    if profile is not None:
        preview = ('<section class="panel" id="results"><h2>Peptide preference motif</h2>'
                   '<p class="muted">Larger letters indicate higher amino-acid probabilities; taller stacks indicate more concentrated positional preferences. This sequence logo represents model predictions, not an experimentally measured binding motif.</p>'
                   + _profile_logo(profile) + '</section>')
    elif table is not None:
        count = table.height if total_rows is None else total_rows
        if workflow in {"pmhc-score", "tcr-score"}:
            contents, displayed_columns = _score_preview(table, columns, background, background_metadata)
            preview = ('<section class="panel" id="results"><h2>Top-ranked peptides</h2>'
                       '<p class="muted">Up to 10 distinct tested peptides per comparable context and peptide length. Full tables retain every input, including duplicate observations and unresolved rows.</p>'
                       + contents + '</section>')
        else:
            preview = ('<section class="panel" id="results"><h2>Result preview</h2>'
                       f'<p class="muted">Showing {min(table.height,200):,} of {count:,} rows. Preview of at most 200 rows; full CSV and typed Parquet files retain the complete results.</p>'
                       + _table(table, columns) + '</section>')
    if embedding_table is not None:
        embedding_columns = [key for key in ("receptor_id", "neighbor_rank", "reference_id",
                             "cosine_distance", "peptide", "hla", "hla_status", "status", "source", "reason")
                             if key in embedding_table.columns]
        embedding_count = embedding_table.height if embedding_total_rows is None else embedding_total_rows
        embedding_results = (f'<p class="muted">Showing {min(embedding_table.height, 200):,} of {embedding_count:,} evidence rows. A reference receptor can have multiple evidence rows. Excluded or unresolved inputs are retained in embedding_audit.csv.</p>'
                             + _table(embedding_table.head(200), embedding_columns)
                             if embedding_count else
                             '<div class="empty">No embedding neighbors were returned. Inspect embedding_audit.csv for missing chains, reconstruction failures or excluded reference contexts. This does not establish that no peptide is recognized.</div>')
        preview += ('<section class="panel" id="embedding-matches"><h2>Experimental embedding neighbors</h2>'
                    '<p>Receptor embeddings are computed without peptide or MHC input. Neighbors are ranked by cosine distance; smaller distances indicate closer embeddings, not a binding probability. Panel and donor MHC compatibility are checked separately. Embedding distances do not alter CDR3 match criteria or assign antigen specificity.</p>'
                    + embedding_results + '</section>')
    guide = _workflow_guide(workflow, background_metadata)
    glossary = _column_guide(displayed_columns) if profile is None else ""
    nav = ('<a href="#interpretation">Interpretation</a><a href="#context">Run context</a>'
           + ('<a href="#profile">Probability heatmap</a>' if profile is not None else '')
           + ('<a href="#results">Motif</a>' if profile is not None else '<a href="#results">Results</a>' if table is not None else '')
           + ('<a href="#columns">Column definitions</a>' if glossary else '') + '<a href="#files">Output files</a>')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>{_escape(title)} · TCR-Workbench</title><style>{_STYLE}</style></head>
<body><header class="masthead"><div class="brand">TCR<span>·</span>Workbench</div></header>
<main><p class="eyebrow">Analysis report</p><h1>{_escape(title)}</h1>
<nav class="nav" aria-label="Report sections">{nav}</nav><div class="cards">{cards}</div>
<section class="note" id="interpretation"><h2>Interpretation</h2><p>{_escape(interpretation)}</p>{reason_html}{guide}</section>
<section class="panel" id="context"><h2>Run context</h2><dl class="metadata">{details}</dl>
<p class="muted">The manifest records software, inputs and any model used. Retain the full output folder when sharing results.</p></section>
{chart}{preview}{glossary}<section class="panel" id="files"><h2>Output files</h2>
<p class="muted">The tables contain the complete results. Keep these files beside report.html so links remain available.</p><div class="files">{links or 'No companion tables are available.'}</div></section>
<footer class="footer">TCR-Workbench {_escape(__version__)} · Offline HTML report; no external scripts or network requests.</footer></main></body></html>'''
    (directory / "report.html").write_text(page, encoding="utf-8")


def write_report(directory: Path, evidence: pl.DataFrame, receptors: pl.DataFrame,
                 qc: list[dict[str, Any]], run: dict[str, Any], *,
                 evidence_summary: dict[str, int] | None = None) -> None:
    candidate_ids = (evidence_summary["receptors_with_candidates"] if evidence_summary else
                     evidence.filter(pl.col("status") == "Candidate")["receptor_id"].n_unique())
    summary = {
        "receptors": receptors.height, "receptors_with_candidates": candidate_ids,
        "receptors_without_candidates": receptors.height - candidate_ids,
        "evidence_rows": evidence_summary["evidence_rows"] if evidence_summary else evidence.height,
        "qc_events": len(qc),
    }
    run["summary"] = summary
    write_json(directory / "qc.json", qc)
    columns = [x for x in ("receptor_id", "status", "peptide", "hla", "evidence_type",
                           "distance", "hla_status", "source", "reason") if x in evidence.columns]
    embedding = run.get("embedding_matching")
    embedding_table = (pl.scan_parquet(directory / "embedding_matches.parquet").head(200).collect()
                       if embedding is not None else None)
    write_workflow_report(directory, "Reference matches", table=evidence, columns=columns,
        summary=summary, total_rows=summary["evidence_rows"], manifest_pending=True,
        embedding_table=embedding_table,
        embedding_total_rows=embedding.get("match_rows") if embedding else None,
        metadata={"Method": "Exact / similarity reference matching" + ("; experimental embedding neighbors" if embedding else ""),
                  "Model": "Used only for experimental embeddings; see embedding_metadata.json" if embedding else "Not used",
                  **({"Reference species filter": run["parameters"]["reference_species_filter"] + " (CDR3 and embedding searches)"}
                     if run.get("parameters", {}).get("reference_species_filter") else {}),
                  "Maximum summed edit distance": run.get("parameters", {}).get("max_distance")},
        interpretation="Candidate denotes a receptor–antigen hypothesis supported by the supplied reference. Exact CDR3 matches, sequence similarity and HLA compatibility do not establish binding. Unresolved means insufficient evidence, not nonbinding. Alternative receptor pairings can share source cells; their cell counts must not be summed as independent observations. Missing chains may reflect experimental dropout, barcode contamination or primer bias.")
    run["outputs"] = {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                      for p in sorted(directory.iterdir()) if p.is_file()}
    write_json(directory / "manifest.json", run)
