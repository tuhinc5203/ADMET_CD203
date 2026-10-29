# hERG blocker predictor - Gradio app serving the Part 1 ONNX model
import importlib.util
import re
from collections import deque
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# NAME, THRESHOLD and featurize come straight from the Part 1 training script (copied next
# to this file), so the app always builds exactly the features the ONNX model was trained on
HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "part1", HERE / "Part1_Olivia-Lu-Suraj-Godithi-Aditi-Shankar-Tuhin-Chatterjee.py")
part1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(part1)
NAME, THRESHOLD, featurize = part1.NAME, part1.THRESHOLD, part1.featurize

SESSION = ort.InferenceSession(str(HERE / f"Part1_{NAME}.onnx"), providers=["CPUExecutionProvider"])
INPUT = SESSION.get_inputs()[0].name

EXAMPLES = [("Terfenadine", "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"),
            ("Haloperidol", "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1"),
            ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1"),
            ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
            ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]


def is_valid(smiles):
    s = smiles.strip()
    if not s:
        return False
    mol = Chem.MolFromSmiles(s)
    return mol is not None and mol.GetNumAtoms() > 0


def validate(smiles):
    # live feedback + toggle the Predict button; clear any result from the previous molecule
    if not smiles.strip():
        status, ok = gr.update(value="Enter a SMILES string to begin.", elem_classes="hint"), False
    elif is_valid(smiles):
        status, ok = gr.update(value="Valid SMILES", elem_classes="ok"), True
    else:
        status, ok = gr.update(value="Not a valid SMILES. Check brackets, ring numbers and atom symbols.",
                               elem_classes="bad"), False
    return status, gr.update(interactive=ok, elem_classes=[]), "", ""


def structure(smiles):
    """2D drawing as SVG, with an animation delay on every bond and atom label so the
    molecule assembles outward from one atom (breadth-first) when it appears."""
    # prepare first (adds Hs at stereocentres) so atom/bond indices match the drawing
    mol = rdMolDraw2D.PrepareMolForDrawing(Chem.MolFromSmiles(smiles))
    drawer = rdMolDraw2D.MolDraw2DSVG(440, 300)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().bondLineWidth = 2
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    step, n = {}, 0
    for start in range(mol.GetNumAtoms()):  # loop covers every fragment of a salt
        if start in step:
            continue
        step[start], queue = n, deque([start])
        while queue:
            atom = mol.GetAtomWithIdx(queue.popleft())
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in step:
                    n += 1
                    step[nbr.GetIdx()] = n
                    queue.append(nbr.GetIdx())
        n += 1
    dt = min(0.12, 1.8 / max(n, 1))  # whole build takes at most ~2 s
    bond_step = {b.GetIdx(): max(step[b.GetBeginAtomIdx()], step[b.GetEndAtomIdx()]) for b in mol.GetBonds()}

    def delay(match):
        element, kind, idx = match.group(0), match.group(1), int(match.group(2))
        t = (bond_step[idx] if kind == "bond" else step[idx] + 0.6) * dt
        if kind == "bond":
            element = element.replace("<path ", "<path pathLength='1' ", 1)
        if "style='" in element:
            return element.replace("style='", f"style='animation-delay:{t:.2f}s;", 1)
        return element.replace("/>", f" style='animation-delay:{t:.2f}s' />", 1)

    # RDKit also draws small unlabelled corner pieces where bonds meet; reveal each with its atom
    coords = [drawer.GetDrawCoords(i) for i in range(mol.GetNumAtoms())]

    def join(match):
        x, y = float(match.group(1)), float(match.group(2))
        atom = min(range(len(coords)), key=lambda i: (coords[i].x - x) ** 2 + (coords[i].y - y) ** 2)
        return match.group(0).replace("<path ", f"<path class='join' style='animation-delay:{(step[atom] + 1) * dt:.2f}s' ", 1)

    svg = re.sub(r"<path class='(bond|atom)-(\d+)[^>]*/>", delay, drawer.GetDrawingText())
    svg = re.sub(r"<path d='M [\d.]+,[\d.]+ L ([\d.]+),([\d.]+)[^>]*/>", join, svg) if coords else svg
    svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
    svg = re.sub(r"width='\d+px' height='\d+px'", "width='100%'", svg, count=1)
    return f"<div class='mol'>{svg}</div>"


def predict(smiles):
    if not is_valid(smiles):
        raise gr.Error("Please enter a valid SMILES string.")
    p = float(np.asarray(SESSION.run(None, {INPUT: featurize([smiles.strip()])})[-1])[0, 1])
    if p >= THRESHOLD:
        result = "blocker"
        text = ("**Likely hERG blocker**\n\nThis molecule may block the hERG potassium channel, which can "
                "disrupt heart rhythm. Experimental hERG testing is recommended.")
    else:
        result = "safe"
        text = "**Unlikely hERG blocker**\n\nThe model does not flag this molecule for hERG blockade."
    return (gr.update(elem_classes=[result]), text + f"\n\nProbability of blocking hERG: {p:.1%}",
            structure(smiles.strip()))


# keep the page in light mode even when the visitor's system prefers dark
FORCE_LIGHT = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.href);
    }
}
"""

CSS = """
footer {display: none !important;}
#wrap {max-width: 640px; margin: 48px auto 0;}
#title h1 {font-weight: 600; letter-spacing: -0.01em; margin-bottom: 4px;}
#title p {color: #6b7280; margin-top: 0;}
#status {min-height: 24px; font-size: 14px;}
.ok, .ok * {color: #15803d !important;} .bad, .bad * {color: #b91c1c !important;}
.hint, .hint * {color: #9ca3af !important;}
#predict {width: 136px; height: 136px; min-width: 136px; max-width: 136px; flex: 0 0 auto !important;
          align-self: center; margin: 20px auto 8px; border-radius: 50%; border: none; box-shadow: none;
          background: #111827; color: #fff; font-size: 17px; font-weight: 500;
          transition: background 0.25s ease, transform 0.1s ease;}
#predict:hover:enabled {transform: scale(1.03);}
#predict:disabled {background: #e5e7eb; color: #9ca3af; cursor: not-allowed; opacity: 1;}
#predict.blocker {background: #dc2626;}
#predict.safe {background: #16a34a;}
#result {text-align: center; max-width: 440px; margin: 0 auto; min-height: 96px;}
#result p {margin: 6px 0;}
#note, #note * {text-align: center; color: #9ca3af !important; margin-top: 32px;}
.mol {display: flex; justify-content: center;}
.mol svg {width: 100%; max-width: 440px; height: auto;}
.mol [class^='bond-'] {stroke-dasharray: 1; stroke-dashoffset: 1; opacity: 0;
                       animation: mol-draw 0.45s ease-out forwards;}
.mol [class^='atom-'] {opacity: 0; transform-box: fill-box; transform-origin: center;
                       animation: mol-pop 0.4s cubic-bezier(0.3, 1.6, 0.6, 1) forwards;}
.mol .join {opacity: 0; animation: mol-fade 0.2s ease-out forwards;}
@keyframes mol-fade {to {opacity: 1;}}
@keyframes mol-draw {0% {opacity: 0;} 15% {opacity: 1;} 100% {opacity: 1; stroke-dashoffset: 0;}}
@keyframes mol-pop {from {opacity: 0; transform: scale(0.3);} to {opacity: 1; transform: scale(1);}}
@media (prefers-reduced-motion: reduce) {
  .mol [class^='bond-'], .mol [class^='atom-'], .mol .join {animation: none; opacity: 1; stroke-dashoffset: 0;}
}
"""

THEME = gr.themes.Base(primary_hue="neutral", neutral_hue="gray", radius_size="md",
                       font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"])

with gr.Blocks(title="hERG Blocker Predictor", css=CSS, theme=THEME, js=FORCE_LIGHT) as demo:
    with gr.Column(elem_id="wrap"):
        gr.Markdown("# hERG Blocker Predictor\nEstimate whether a molecule blocks the hERG potassium channel, "
                    "a common cause of drug-induced cardiotoxicity.", elem_id="title")
        with gr.Row(equal_height=True):
            smiles = gr.Textbox(label="SMILES", placeholder="Paste a SMILES string", max_lines=1, scale=3,
                                min_width=300)
            examples = gr.Dropdown([name for name, _ in EXAMPLES], value=None, label="Examples", scale=1,
                                   min_width=160)
        status = gr.Markdown("Enter a SMILES string to begin.", elem_id="status", elem_classes="hint")
        button = gr.Button("Predict", elem_id="predict", interactive=False)
        result = gr.Markdown(elem_id="result")
        molecule = gr.HTML(elem_id="structure")
        gr.Markdown("<sub>Extra-trees model on Morgan fingerprints and RDKit descriptors. "
                    "For research use only.</sub>", elem_id="note")

    examples.input(lambda name: dict(EXAMPLES).get(name, gr.update()), examples, smiles)
    smiles.change(validate, smiles, [status, button, result, molecule], show_progress="hidden", queue=False)
    button.click(predict, smiles, [button, result, molecule], show_progress="hidden")
    smiles.submit(predict, smiles, [button, result, molecule], show_progress="hidden")

if __name__ == "__main__":
    demo.launch()
