# hERG blocker predictor - Gradio app serving the Part 1 ONNX model
import importlib.util
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from rdkit import Chem

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

EXAMPLES = [["CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"],  # terfenadine
            ["CC(=O)Oc1ccccc1C(=O)O"],  # aspirin
            ["CN(C)CCOC(c1ccccc1)c1ccccc1"]]  # diphenhydramine


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
        status, ok = gr.update(value="✓ Valid SMILES", elem_classes="ok"), True
    else:
        status, ok = gr.update(value="✗ Not a valid SMILES - check brackets, ring numbers and atom symbols.",
                               elem_classes="bad"), False
    return status, gr.update(interactive=ok), ""


def predict(smiles):
    if not is_valid(smiles):
        raise gr.Error("Please enter a valid SMILES string.")
    p = float(np.asarray(SESSION.run(None, {INPUT: featurize([smiles.strip()])})[-1])[0, 1])
    if p >= THRESHOLD:
        verdict = ("### ⚠️ Likely hERG blocker\nThis molecule may carry cardiotoxicity risk; "
                   "consider experimental hERG testing.")
    else:
        verdict = "### ✅ Unlikely hERG blocker\nThe model does not flag this molecule."
    return verdict + f"\n\nProbability of blocking hERG: **{p:.1%}** (flagged at {THRESHOLD:.0%} or higher)"


CSS = """
.ok {color: #15803d;} .bad {color: #b91c1c;} .hint {color: #6b7280;}
#wrap {max-width: 720px; margin: 0 auto;}
"""

with gr.Blocks(title="hERG Blocker Predictor", css=CSS, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="wrap"):
        gr.Markdown("# hERG Blocker Predictor\nPaste a molecule's SMILES string to estimate whether it blocks the "
                    "hERG potassium channel, a common cause of drug-induced cardiotoxicity.")
        smiles = gr.Textbox(label="SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O", max_lines=1)
        status = gr.Markdown("Enter a SMILES string to begin.", elem_classes="hint")
        button = gr.Button("Predict", variant="primary", interactive=False)
        verdict = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=smiles, label="Try an example")
        gr.Markdown(f"<sub>Extra-trees classifier on Morgan fingerprints + physicochemical descriptors. "
                    f"Flags a blocker when P(blocker) ≥ {THRESHOLD}. For research use only.</sub>")

    smiles.change(validate, smiles, [status, button, verdict], show_progress="hidden", queue=False)
    button.click(predict, smiles, verdict)
    smiles.submit(predict, smiles, verdict)

if __name__ == "__main__":
    demo.launch()
