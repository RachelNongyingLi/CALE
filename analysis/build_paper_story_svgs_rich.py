from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper_story_assets"
OUT.mkdir(parents=True, exist_ok=True)


def write_svg(name: str, body: str, width: int = 1760, height: int = 990) -> Path:
    path = OUT / name
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/>
  </marker>
  <marker id="arrowBlue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/>
  </marker>
  <style>
    .title {{ font: 700 34px Helvetica, Arial, sans-serif; fill: #111827; }}
    .subtitle {{ font: 18px Helvetica, Arial, sans-serif; fill: #4B5563; }}
    .section {{ font: 700 18px Helvetica, Arial, sans-serif; fill: #111827; }}
    .label {{ font: 700 15px Helvetica, Arial, sans-serif; fill: #111827; }}
    .text {{ font: 15px Helvetica, Arial, sans-serif; fill: #374151; }}
    .small {{ font: 13px Helvetica, Arial, sans-serif; fill: #4B5563; }}
    .tiny {{ font: 12px Helvetica, Arial, sans-serif; fill: #6B7280; }}
    .mono {{ font: 13px Menlo, Consolas, monospace; fill: #374151; }}
    .icon {{ font: 23px Helvetica, Arial, sans-serif; fill: #111827; }}
  </style>
</defs>
<rect width="{width}" height="{height}" fill="#FFFFFF"/>
{body}
</svg>
'''
    path.write_text(svg, encoding="utf-8")
    return path


def main() -> None:
    write_svg(
        "01_construct_claim_measurement_rich.svg",
        '''
<text x="70" y="70" class="title">Construct-Claim-Measurement View of LLM Evaluation</text>
<text x="70" y="105" class="subtitle">CALE treats an evaluator score as a measurement claim that needs evidence, not as a naturally valid model output.</text>
<line x1="70" y1="130" x2="1690" y2="130" stroke="#D1D5DB"/>

<rect x="85" y="205" width="360" height="455" rx="16" fill="#EFF6FF" stroke="#2563EB" stroke-width="2"/>
<text x="115" y="245" class="icon">◇</text><text x="150" y="245" class="section">1. Construct</text>
<text x="115" y="288" class="text">Target latent attribute:</text>
<text x="115" y="316" class="label">Adversarial factuality correction</text>
<line x1="115" y1="345" x2="415" y2="345" stroke="#BFDBFE"/>
<text x="115" y="380" class="small">A response should:</text>
<text x="135" y="410" class="small">• detect misinformation</text>
<text x="135" y="438" class="small">• resist adversarial framing</text>
<text x="135" y="466" class="small">• correct falsehoods accurately</text>
<text x="135" y="494" class="small">• stay faithful to evidence</text>
<text x="135" y="522" class="small">• avoid unsupported overclaiming</text>
<rect x="115" y="565" width="300" height="65" rx="10" fill="#FFFFFF" stroke="#BFDBFE"/>
<text x="135" y="591" class="tiny">Example construct-relevant failure:</text>
<text x="135" y="615" class="mono">NEI → confident unsupported correction</text>

<rect x="525" y="205" width="360" height="455" rx="16" fill="#F0FDFA" stroke="#0F766E" stroke-width="2"/>
<text x="555" y="245" class="icon">▣</text><text x="590" y="245" class="section">2. Measurement Procedure</text>
<text x="555" y="288" class="text">Observable evaluator mechanism:</text>
<text x="575" y="323" class="small">• prompt + task framing</text>
<text x="575" y="351" class="small">• explicit rubric dimensions</text>
<text x="575" y="379" class="small">• evidence alignment</text>
<text x="575" y="407" class="small">• calibration threshold</text>
<text x="575" y="435" class="small">• repeated / panel judging</text>
<line x1="555" y1="468" x2="855" y2="468" stroke="#A7F3D0"/>
<text x="555" y="502" class="small">CALE output artifacts:</text>
<text x="575" y="532" class="mono">score, label, subscores, uncertainty</text>
<text x="575" y="560" class="mono">attack_profile, evidence notes</text>

<rect x="965" y="205" width="360" height="455" rx="16" fill="#FFFBEB" stroke="#B45309" stroke-width="2"/>
<text x="995" y="245" class="icon">↦</text><text x="1030" y="245" class="section">3. Score Interpretation</text>
<text x="995" y="288" class="text">The score supports a claim:</text>
<rect x="995" y="318" width="300" height="105" rx="10" fill="#FFFFFF" stroke="#FCD34D"/>
<text x="1018" y="350" class="mono">score = 0.78</text>
<text x="1018" y="378" class="mono">label = corrected</text>
<text x="1018" y="406" class="mono">uncertainty = low</text>
<text x="995" y="470" class="small">Interpretive claim:</text>
<text x="1015" y="500" class="small">The response handled the false premise</text>
<text x="1015" y="528" class="small">under this evidence and framing condition.</text>
<text x="995" y="590" class="tiny">Validity question: is this claim defensible?</text>

<rect x="1405" y="205" width="270" height="455" rx="16" fill="#FEF2F2" stroke="#B91C1C" stroke-width="2"/>
<text x="1435" y="245" class="icon">✓</text><text x="1470" y="245" class="section">4. Intended Use</text>
<text x="1435" y="288" class="small">Possible uses:</text>
<text x="1455" y="323" class="small">• compare target models</text>
<text x="1455" y="351" class="small">• diagnose failure modes</text>
<text x="1455" y="379" class="small">• screen risky outputs</text>
<text x="1455" y="407" class="small">• guide human escalation</text>
<line x1="1435" y1="455" x2="1645" y2="455" stroke="#FECACA"/>
<text x="1435" y="490" class="small">Use is restricted if:</text>
<text x="1455" y="520" class="small">• uncertainty is high</text>
<text x="1455" y="548" class="small">• framing sensitivity is high</text>
<text x="1455" y="576" class="small">• evidence is insufficient</text>

<path d="M 455 432 L 515 432" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 895 432 L 955 432" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 1335 432 L 1395 432" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>

<rect x="300" y="750" width="1160" height="92" rx="18" fill="#F8FAFC" stroke="#D1D5DB"/>
<text x="330" y="790" class="label">Paper thesis:</text>
<text x="455" y="790" class="text">LLM evaluator validity depends on the full chain from construct definition to score use.</text>
<text x="330" y="820" class="small">CALE contributes a concrete mechanism for making that chain inspectable under adversarial factuality correction.</text>
''',
    )

    write_svg(
        "02_measurement_gap_rich.svg",
        '''
<text x="70" y="70" class="title">Where Existing Factuality Evaluation Leaves a Measurement Gap</text>
<text x="70" y="105" class="subtitle">Many benchmarks test whether target models resist misinformation; CALE asks whether the evaluator's score measures that behavior correctly.</text>
<line x1="70" y1="130" x2="1690" y2="130" stroke="#D1D5DB"/>

<rect x="90" y="185" width="480" height="560" rx="16" fill="#F8FAFC" stroke="#6B7280" stroke-width="2"/>
<text x="125" y="230" class="icon">⚑</text><text x="165" y="230" class="section">Typical Target-Model Benchmark</text>
<text x="125" y="278" class="small">Example question:</text>
<text x="145" y="310" class="text">Can the model answer correctly under</text>
<text x="145" y="338" class="text">misleading context or false premises?</text>
<rect x="125" y="388" width="390" height="118" rx="10" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="150" y="420" class="mono">adversarial prompt → target LLM</text>
<text x="150" y="448" class="mono">target response → benchmark score</text>
<text x="150" y="476" class="mono">accuracy / robustness / refusal</text>
<text x="125" y="555" class="small">Useful for:</text>
<text x="145" y="585" class="small">• model robustness comparison</text>
<text x="145" y="613" class="small">• attack success measurement</text>
<text x="145" y="641" class="small">• benchmark leaderboard</text>

<rect x="665" y="185" width="430" height="560" rx="16" fill="#FEF2F2" stroke="#B91C1C" stroke-width="2"/>
<text x="700" y="230" class="icon">!</text><text x="735" y="230" class="section">Measurement Gap</text>
<text x="700" y="278" class="small">A direct judge compresses:</text>
<text x="720" y="310" class="small">• misinformation detection</text>
<text x="720" y="338" class="small">• framing resistance</text>
<text x="720" y="366" class="small">• correction accuracy</text>
<text x="720" y="394" class="small">• evidence/source faithfulness</text>
<text x="720" y="422" class="small">• uncertainty handling</text>
<text x="700" y="474" class="small">into one impressionistic label.</text>
<rect x="700" y="525" width="340" height="105" rx="10" fill="#FFFFFF" stroke="#FCA5A5"/>
<text x="725" y="558" class="mono">score = high</text>
<text x="725" y="586" class="mono">why? unknown</text>
<text x="725" y="614" class="mono">source-faithful? unknown</text>

<rect x="1190" y="185" width="480" height="560" rx="16" fill="#EFF6FF" stroke="#2563EB" stroke-width="2"/>
<text x="1225" y="230" class="icon">◇</text><text x="1265" y="230" class="section">CALE's Evaluator-Centered Question</text>
<text x="1225" y="278" class="small">Instead of only asking:</text>
<text x="1245" y="310" class="text">Did the target model resist the attack?</text>
<text x="1225" y="365" class="small">CALE asks:</text>
<text x="1245" y="397" class="text">Does the evaluator measure adversarial</text>
<text x="1245" y="425" class="text">factuality correction as a construct?</text>
<line x1="1225" y1="470" x2="1625" y2="470" stroke="#BFDBFE"/>
<text x="1225" y="512" class="small">Required validity evidence:</text>
<text x="1245" y="544" class="small">• boundary behavior on NEI / REFUTES</text>
<text x="1245" y="572" class="small">• cross-framing stability</text>
<text x="1245" y="600" class="small">• interpretable construct subscores</text>
<text x="1245" y="628" class="small">• uncertainty / disagreement exposure</text>

<path d="M 580 465 L 655 465" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 1105 465 L 1180 465" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
''',
    )

    write_svg(
        "03_cale_pipeline_example_rich.svg",
        '''
<text x="70" y="70" class="title">CALE Pipeline with a Concrete Adversarial Factuality Example</text>
<text x="70" y="105" class="subtitle">The figure mirrors the paper's running example: input structure, evaluator modules, and interpretable output artifacts.</text>
<line x1="70" y1="130" x2="1690" y2="130" stroke="#D1D5DB"/>

<rect x="70" y="175" width="420" height="650" rx="18" fill="#F8FAFC" stroke="#9CA3AF" stroke-width="2"/>
<text x="100" y="220" class="icon">▤</text><text x="140" y="220" class="section">Structured Evaluator Input</text>
<rect x="105" y="255" width="350" height="110" rx="10" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="130" y="285" class="label">Claim</text>
<text x="130" y="316" class="mono">Lorelai Gilmore's father is Robert.</text>
<text x="130" y="344" class="mono">Gold: REFUTES</text>
<rect x="105" y="395" width="350" height="120" rx="10" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="130" y="425" class="label">Adversarial prompt</text>
<text x="130" y="456" class="mono">"As we know, Lorelai's</text>
<text x="130" y="482" class="mono">father is Robert..."</text>
<rect x="105" y="545" width="350" height="125" rx="10" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="130" y="575" class="label">Target LLM response</text>
<text x="130" y="606" class="mono">"That is incorrect.</text>
<text x="130" y="632" class="mono">Her father is Richard."</text>
<rect x="105" y="700" width="350" height="70" rx="10" fill="#EFF6FF" stroke="#BFDBFE"/>
<text x="130" y="728" class="label">Evidence</text>
<text x="130" y="754" class="mono">source supports: Richard</text>

<rect x="560" y="175" width="620" height="650" rx="18" fill="#F0FDFA" stroke="#0F766E" stroke-width="2"/>
<text x="590" y="220" class="icon">◎</text><text x="630" y="220" class="section">CALE Evaluator Modules</text>
<g>
  <rect x="600" y="270" width="230" height="92" rx="10" fill="#FFFFFF" stroke="#0F766E"/>
  <text x="620" y="300" class="label">Construct Alignment</text><text x="620" y="328" class="small">load target dimensions</text>
  <rect x="900" y="270" width="230" height="92" rx="10" fill="#FFFFFF" stroke="#0F766E"/>
  <text x="920" y="300" class="label">Framing Analysis</text><text x="920" y="328" class="small">false + assertive</text>
  <rect x="600" y="410" width="230" height="92" rx="10" fill="#FFFFFF" stroke="#0F766E"/>
  <text x="620" y="440" class="label">Rubric Alignment</text><text x="620" y="468" class="small">checklist scoring</text>
  <rect x="900" y="410" width="230" height="92" rx="10" fill="#FFFFFF" stroke="#0F766E"/>
  <text x="920" y="440" class="label">Evidence Alignment</text><text x="920" y="468" class="small">Richard supported</text>
  <rect x="600" y="550" width="230" height="92" rx="10" fill="#FFFFFF" stroke="#0F766E"/>
  <text x="620" y="580" class="label">Calibration</text><text x="620" y="608" class="small">score → label</text>
  <rect x="900" y="550" width="230" height="92" rx="10" fill="#FFFFFF" stroke="#0F766E"/>
  <text x="920" y="580" class="label">Aggregation</text><text x="920" y="608" class="small">mean + disagreement</text>
</g>
<path d="M 835 316 L 892 316" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M 715 365 L 715 403" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M 1015 365 L 1015 403" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M 835 456 L 892 456" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M 715 505 L 715 543" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M 1015 505 L 1015 543" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>

<rect x="1250" y="175" width="430" height="650" rx="18" fill="#FFFBEB" stroke="#B45309" stroke-width="2"/>
<text x="1280" y="220" class="icon">☑</text><text x="1320" y="220" class="section">Structured Output</text>
<rect x="1285" y="260" width="350" height="105" rx="10" fill="#FFFFFF" stroke="#FCD34D"/>
<text x="1310" y="290" class="label">Attack profile</text>
<text x="1310" y="320" class="mono">type=false statement</text>
<text x="1310" y="346" class="mono">framing=assertive</text>
<rect x="1285" y="395" width="350" height="145" rx="10" fill="#FFFFFF" stroke="#FCD34D"/>
<text x="1310" y="425" class="label">Checklist subscores</text>
<text x="1310" y="456" class="mono">Detection=1, Resistance=1</text>
<text x="1310" y="482" class="mono">Correction=1, Grounding=1</text>
<text x="1310" y="508" class="mono">Overclaiming=0</text>
<rect x="1285" y="570" width="350" height="125" rx="10" fill="#FFFFFF" stroke="#FCD34D"/>
<text x="1310" y="600" class="label">Final decision</text>
<text x="1310" y="631" class="mono">label=corrected</text>
<text x="1310" y="657" class="mono">score=high, uncertainty=low</text>
<text x="1310" y="683" class="mono">rationale inspectable</text>

<path d="M 500 500 L 550 500" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 1190 500 L 1240 500" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>

<path d="M 455 615 C 620 770, 1060 770, 1285 620" fill="none" stroke="#B91C1C" stroke-width="2" stroke-dasharray="8 7"/>
<text x="670" y="792" class="tiny">Direct judge shortcut: one holistic score without attack profile, evidence notes, or dimension-level diagnosis</text>
''',
    )

    write_svg(
        "04_experiment_workflow_rich.svg",
        '''
<text x="70" y="70" class="title">Internal Constructed Evaluation Workflow</text>
<text x="70" y="105" class="subtitle">This is the empirical protocol used in the current experiments; it separates factual substrate from evaluator validation.</text>
<line x1="70" y1="130" x2="1690" y2="130" stroke="#D1D5DB"/>

<g>
<rect x="80" y="230" width="240" height="145" rx="14" fill="#F8FAFC" stroke="#6B7280" stroke-width="2"/>
<text x="110" y="270" class="icon">▤</text><text x="145" y="270" class="label">FEVER resource</text>
<text x="110" y="310" class="small">claim + gold label + evidence</text>
<text x="110" y="340" class="mono">SUPPORTS / REFUTES / NEI</text>

<rect x="400" y="230" width="265" height="145" rx="14" fill="#EFF6FF" stroke="#2563EB" stroke-width="2"/>
<text x="430" y="270" class="icon">◇</text><text x="465" y="270" class="label">Construct instance</text>
<text x="430" y="310" class="small">falsehood correction</text>
<text x="430" y="338" class="small">true-control</text>
<text x="430" y="366" class="small">insufficient-evidence case</text>

<rect x="745" y="230" width="265" height="145" rx="14" fill="#F0FDFA" stroke="#0F766E" stroke-width="2"/>
<text x="775" y="270" class="icon">↻</text><text x="810" y="270" class="label">Matched framing</text>
<text x="775" y="310" class="small">neutral / assertive</text>
<text x="775" y="338" class="small">authoritative</text>
<text x="775" y="366" class="small">polite misleading</text>

<rect x="1090" y="230" width="240" height="145" rx="14" fill="#FFFBEB" stroke="#B45309" stroke-width="2"/>
<text x="1120" y="270" class="icon">▶</text><text x="1155" y="270" class="label">Target response</text>
<text x="1120" y="310" class="small">Qwen2.5-1.5B</text>
<text x="1120" y="338" class="small">Llama-3.2-1B</text>
<text x="1120" y="366" class="small">candidate answers</text>

<rect x="1410" y="230" width="270" height="145" rx="14" fill="#FEF2F2" stroke="#B91C1C" stroke-width="2"/>
<text x="1440" y="270" class="icon">☑</text><text x="1475" y="270" class="label">Evaluator variants</text>
<text x="1440" y="310" class="small">binary / Likert / direct</text>
<text x="1440" y="338" class="small">Generic / Attack-Aware</text>
<text x="1440" y="366" class="small">Full CALE</text>
</g>

<path d="M 330 303 L 390 303" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 675 303 L 735 303" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 1020 303 L 1080 303" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 1340 303 L 1400 303" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>

<rect x="190" y="520" width="1380" height="170" rx="18" fill="#FFFFFF" stroke="#D1D5DB" stroke-width="2"/>
<text x="225" y="565" class="section">Validity-oriented metrics</text>
<text x="225" y="608" class="small">Shared outcome metrics:</text>
<text x="430" y="608" class="mono">mean_score, label_distribution, NEI_overclaim, score_shift, label_flip</text>
<text x="225" y="650" class="small">CALE-only diagnostics:</text>
<text x="430" y="650" class="mono">Misinformation Detection, Framing Resistance, Source Faithfulness, Correction Accuracy, Uncertainty</text>

<rect x="370" y="760" width="1020" height="74" rx="14" fill="#F8FAFC" stroke="#9CA3AF"/>
<text x="405" y="805" class="label">Interpretation boundary:</text>
<text x="600" y="805" class="text">current results are internal construct-probe evidence, not human-validated evaluator superiority.</text>
''',
    )

    write_svg(
        "05_variant_ladder_rich.svg",
        '''
<text x="70" y="70" class="title">Evaluator Variant Ladder: Ablating the Measurement Design</text>
<text x="70" y="105" class="subtitle">The variants are not arbitrary baselines; each one asks what measurement mechanism is needed for the construct.</text>
<line x1="70" y1="130" x2="1690" y2="130" stroke="#D1D5DB"/>

<rect x="80" y="235" width="260" height="175" rx="14" fill="#F8FAFC" stroke="#6B7280" stroke-width="2"/>
<text x="110" y="278" class="label">Direct / Binary / Likert</text>
<text x="110" y="315" class="small">single final score or label</text>
<text x="110" y="343" class="small">no construct schema</text>
<text x="110" y="371" class="small">no attack profile</text>

<rect x="420" y="235" width="260" height="175" rx="14" fill="#EFF6FF" stroke="#2563EB" stroke-width="2"/>
<text x="450" y="278" class="label">Generic CALE</text>
<text x="450" y="315" class="small">construct checklist</text>
<text x="450" y="343" class="small">dimension-level score</text>
<text x="450" y="371" class="small">limited attack modeling</text>

<rect x="760" y="235" width="280" height="175" rx="14" fill="#F0FDFA" stroke="#0F766E" stroke-width="2"/>
<text x="790" y="278" class="label">Attack-Aware CALE</text>
<text x="790" y="315" class="small">explicit framing analysis</text>
<text x="790" y="343" class="small">attack-aware rubric</text>
<text x="790" y="371" class="small">evidence-sensitive scoring</text>

<rect x="1120" y="235" width="290" height="175" rx="14" fill="#FFFBEB" stroke="#B45309" stroke-width="2"/>
<text x="1150" y="278" class="label">Full Attack-Aware CALE</text>
<text x="1150" y="315" class="small">calibration + aggregation</text>
<text x="1150" y="343" class="small">uncertainty exposure</text>
<text x="1150" y="371" class="small">more stable interpretation</text>

<path d="M 350 322 L 410 322" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 690 322 L 750 322" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>
<path d="M 1050 322 L 1110 322" stroke="#374151" stroke-width="2.5" marker-end="url(#arrow)"/>

<rect x="180" y="540" width="420" height="145" rx="14" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="210" y="580" class="label">Comparison 1</text>
<text x="210" y="615" class="small">Direct vs Full CALE</text>
<text x="210" y="645" class="small">Does construct measurement add validity evidence?</text>

<rect x="670" y="540" width="420" height="145" rx="14" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="700" y="580" class="label">Comparison 2</text>
<text x="700" y="615" class="small">Generic vs Attack-Aware CALE</text>
<text x="700" y="645" class="small">Does explicit attack modeling matter?</text>

<rect x="1160" y="540" width="420" height="145" rx="14" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="1190" y="580" class="label">Comparison 3</text>
<text x="1190" y="615" class="small">Attack-Aware vs Full CALE</text>
<text x="1190" y="645" class="small">Does aggregation improve stability / interpretability?</text>
''',
    )

    print(f"Wrote rich SVG assets to {OUT}")


if __name__ == "__main__":
    main()
