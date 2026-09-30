"""
=================================================================================
 Emotion-Aware Collections & Service Calls  |  Streamlit POC
=================================================================================
A proof-of-concept that records a short call snippet in the browser, extracts
prosodic (voice) features locally, and sends the audio to a Gemini model for
real-time stress / emotion analysis + agent guidance.

Run:
    pip install -r requirements.txt
    streamlit run emotion_aware_calls_poc.py

Get a free Gemini API key at: https://aistudio.google.com/apikey

NOTE ON MODELS
--------------
Google renames / retires Gemini models regularly. The dropdown below only
lists models that (a) accept native audio input and (b) were live on the
Gemini API as of this build. If a call fails with a 404 / "model not found"
error, open https://ai.google.dev/gemini-api/docs/models and swap the model
id in MODEL_CATALOG for a current one — the rest of the app needs no changes.
=================================================================================
"""

import io
import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- Optional heavy deps: app must not crash if these are missing ----------
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False


# =================================================================================
# 1. CONFIG & THEME
# =================================================================================

st.set_page_config(
    page_title="Emotion-Aware Call Monitor",
    page_icon="\U0001F399\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B3D59"
NAVY_DEEP = "#082C41"
TEAL = "#1C7293"
AMBER = "#E8871E"
GREEN = "#2E9E63"
RED = "#D64545"
INK = "#1B2430"
MUTED = "#5B6B79"
OFFWHITE = "#F7F9FA"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {OFFWHITE};
    }}
    .hero-banner {{
        background: {NAVY};
        background-image: radial-gradient(circle at 85% -10%, {TEAL}55, transparent 55%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
    }}
    .hero-title {{
        color: white;
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}
    .hero-sub {{
        color: #CADCFC;
        font-size: 1.0rem;
    }}
    .kicker {{
        color: {AMBER};
        letter-spacing: 2px;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
    }}
    .poc-card {{
        background: white;
        border: 1px solid #E3E8EC;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        color: white;
    }}
    .escalate-banner {{
        background: {RED};
        color: white;
        padding: 1rem 1.3rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }}
    .ok-banner {{
        background: {GREEN};
        color: white;
        padding: 0.7rem 1.3rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }}
    .footer-note {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: 2rem;
    }}
    div[data-testid="stMetricValue"] {{
        color: {NAVY};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =================================================================================
# 2. MODEL CATALOG  ---  only Gemini models with native AUDIO input support
# =================================================================================
# speed_tier: "fastest" | "fast" | "deep"  (rough latency expectation, not a spec)

MODEL_CATALOG = {
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "speed_tier": "fast",
        "notes": "Solid, still-live baseline. Good for comparison against newer generations.",
    },
    "gemini-2.5-flash-lite": {
        "label": "Gemini 2.5 Flash-Lite",
        "speed_tier": "fastest",
        "notes": "Cheapest of the 2.5 generation — good low-cost baseline.",
    },
    "gemini-3.1-flash": {
        "label": "Gemini 3.1 Flash  (Recommended)",
        "speed_tier": "fast",
        "notes": "Best balance of speed + reasoning quality for live call analysis.",
    },
    "gemini-3.1-flash-lite": {
        "label": "Gemini 3.1 Flash-Lite",
        "speed_tier": "fastest",
        "notes": "Low latency & cost, quality on par with 2.5 Flash — good for high call-volume.",
    },
    "gemini-3.6-flash": {
        "label": "Gemini 3.6 Flash  (Latest)",
        "speed_tier": "fast",
        "notes": "Newest GA flash model — stronger reasoning, lower token usage.",
    },
    "gemini-3.5-flash-lite": {
        "label": "Gemini 3.5 Flash-Lite  (Latest)",
        "speed_tier": "fastest",
        "notes": "Newest, fastest, lowest-cost model — best for high-throughput real-time agent-assist.",
    },
    "gemini-flash-latest": {
        "label": "gemini-flash-latest  (Rolling alias)",
        "speed_tier": "fast",
        "notes": "Always points to Google's current default Flash model — auto-updates as new versions ship, no code change needed.",
    },
    "gemini-flash-lite-latest": {
        "label": "gemini-flash-lite-latest  (Rolling alias)",
        "speed_tier": "fastest",
        "notes": "Always points to Google's current default Flash-Lite model — auto-updates as new versions ship, no code change needed.",
    },
}
DEFAULT_MODEL = "gemini-flash-lite-latest"


# =================================================================================
# 3. SESSION STATE
# =================================================================================

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, one per analyzed call turn
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# =================================================================================
# 4. AUDIO FEATURE EXTRACTION (local, model-free — used for the waveform panel
#    and as an offline fallback when no API key is supplied)
# =================================================================================

def extract_prosody_features(audio_bytes: bytes):
    """Returns (samples, sample_rate, features_dict) or (None, None, None) on failure."""
    if not (HAS_SOUNDFILE and HAS_LIBROSA):
        return None, None, None
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if len(data) < sr * 0.3:  # too short to analyze meaningfully
            return data, sr, None

        rms = librosa.feature.rms(y=data)[0]
        zcr = librosa.feature.zero_crossing_rate(y=data)[0]

        try:
            f0, voiced_flag, _ = librosa.pyin(
                data, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
            )
            f0_voiced = f0[~np.isnan(f0)]
        except Exception:
            f0_voiced = np.array([])

        features = {
            "duration_sec": round(len(data) / sr, 2),
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "pitch_mean_hz": float(np.mean(f0_voiced)) if len(f0_voiced) else None,
            "pitch_std_hz": float(np.std(f0_voiced)) if len(f0_voiced) else None,
            "zero_crossing_rate": float(np.mean(zcr)),
        }
        return data, sr, features
    except Exception:
        return None, None, None


def heuristic_stress_score(features: dict) -> float:
    """Very rough offline fallback — used only when no API key is present."""
    if not features:
        return 0.0
    score = 0.0
    if features.get("pitch_std_hz"):
        score += min(features["pitch_std_hz"] / 40.0, 4.0)
    score += min(features["energy_std"] * 40.0, 3.0)
    score += min(features["zero_crossing_rate"] * 20.0, 3.0)
    return round(min(score, 10.0), 1)


def clamp10(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 1)


def format_features_for_prompt(features: dict) -> str:
    """Renders the locally extracted prosodic features as a short, readable
    block for the Gemini prompt — gives the model an objective numeric anchor
    alongside its own listening, rather than relying on audio alone."""
    if not features:
        return "(No local acoustic features available for this clip \u2014 extraction failed or the clip was too short. Rely on the audio alone.)"

    def fmt(v, suffix="", digits=1):
        return "not detected" if v is None else f"{v:.{digits}f}{suffix}"

    return (
        f"- Clip duration: {fmt(features.get('duration_sec'), ' sec', 2)}\n"
        f"- Mean pitch (F0): {fmt(features.get('pitch_mean_hz'), ' Hz')}\n"
        f"- Pitch variability (std dev): {fmt(features.get('pitch_std_hz'), ' Hz')} "
        f"(higher = more pitch movement/agitation, lower = flatter/steadier tone)\n"
        f"- Mean energy (RMS loudness): {fmt(features.get('energy_mean'), '', 4)}\n"
        f"- Energy variability (std dev): {fmt(features.get('energy_std'), '', 4)} "
        f"(higher = more volume swings)\n"
        f"- Zero-crossing rate: {fmt(features.get('zero_crossing_rate'), '', 4)} "
        f"(higher can indicate harsher/noisier or more sibilant vocal texture)"
    )


def offline_acoustic_table(features: dict):
    """Rough, purely-local approximation of the acoustic scorecard Gemini would
    normally return — used only when no API key is supplied. These are crude
    signal-processing proxies, not a substitute for the model's judgement."""
    if not features:
        return []

    pitch_std = features.get("pitch_std_hz") or 0.0
    energy_std = features.get("energy_std") or 0.0
    energy_mean = features.get("energy_mean") or 0.0
    zcr = features.get("zero_crossing_rate") or 0.0

    return [
        {
            "characteristic": "Pitch stability",
            "score": clamp10(10 - pitch_std / 8.0),
            "observation": "Lower pitch variance reads as a steadier tone.",
        },
        {
            "characteristic": "Volume consistency",
            "score": clamp10(10 - energy_std * 60.0),
            "observation": "Steadier RMS energy across the clip scores higher.",
        },
        {
            "characteristic": "Overall loudness",
            "score": clamp10(energy_mean * 60.0),
            "observation": "Relative energy level of the clip.",
        },
        {
            "characteristic": "Sound clarity (proxy)",
            "score": clamp10(10 - zcr * 20.0),
            "observation": "Zero-crossing based proxy only.",
        },
        {
            "characteristic": "Voice texture (proxy)",
            "score": clamp10(10 - zcr * 30.0),
            "observation": "Rough jitter-like estimate — no true harmonic analysis.",
        },
    ]


def offline_in_depth_analysis(features: dict, score: float) -> dict:
    """Builds a structured, analytically-honest explanation from the raw local
    numbers, mirroring the schema Gemini would return, so offline mode is also
    fully inspectable rather than a black-box number."""
    if not features:
        return {
            "confidence": 0.0,
            "language_confidence": 0.0,
            "key_evidence": [],
            "stress_score_reasoning": "No usable audio features were extracted from this clip.",
            "emotion_breakdown_reasoning": "Not available.",
            "spoken_content_analysis": "Not available \u2014 offline mode has no speech-to-text or language understanding.",
            "context_influence": "Not available \u2014 offline mode does not use account context.",
            "alternative_explanations": "Clip may have been too short, silent, or an unsupported format.",
            "limitations": "No features means no basis for any read at all.",
        }

    pitch_std = features.get("pitch_std_hz")
    energy_std = features.get("energy_std") or 0.0
    zcr = features.get("zero_crossing_rate") or 0.0

    evidence = []
    if pitch_std is not None:
        if pitch_std > 35:
            evidence.append(f"pitch varied by roughly \u00b1{pitch_std:.0f} Hz across the clip \u2014 wide swings usually track agitation")
        else:
            evidence.append(f"pitch stayed within roughly \u00b1{pitch_std:.0f} Hz \u2014 a narrow range usually tracks a calmer tone")
    else:
        evidence.append("pitch could not be reliably tracked (clip may be too short or too noisy)")

    if energy_std > 0.03:
        evidence.append("loudness (RMS energy) varied noticeably turn-to-turn, suggesting sudden emphasis or raised volume")
    else:
        evidence.append("loudness stayed fairly even across the clip")

    if zcr > 0.08:
        evidence.append("a higher zero-crossing rate suggests a harsher, more sibilant, or strained vocal texture")
    else:
        evidence.append("zero-crossing rate was in a typical range for relaxed speech")

    return {
        "confidence": 0.35,
        "language_confidence": 0.0,
        "key_evidence": evidence,
        "stress_score_reasoning": (
            f"Combining pitch variance, loudness variance and zero-crossing rate with fixed "
            f"weights produced a stress score of {score}/10. This is a linear heuristic, not "
            f"a learned or contextual judgement."
        ),
        "emotion_breakdown_reasoning": "Emotion proportions are a fixed function of the stress score only \u2014 offline mode cannot distinguish frustration from anxiety from anger.",
        "spoken_content_analysis": "Not available \u2014 offline mode has no speech-to-text or language understanding. Add a Gemini API key for language detection, transcript, and translation.",
        "context_influence": "None \u2014 offline heuristic mode ignores account context (DPD, segment, prior complaints) entirely.",
        "alternative_explanations": "Poor microphone/line quality, a naturally fast or breathy speaker, or background noise could all produce the same acoustic numbers without any real distress.",
        "limitations": "No semantic understanding of words, no comparison to this caller's normal voice, and no model reasoning \u2014 purely signal-processing thresholds. Add a Gemini API key for a real contextual read.",
    }


# =================================================================================
# 5. GEMINI CALL
# =================================================================================

RESPONSE_SCHEMA_HINT = """
    Return ONLY a single valid JSON object — no markdown fences, no commentary — with
    exactly these fields:
    {
        "escalate_to_human": <true/false>,
        "stress_score": <float 0-10, 0=calm 10=extreme distress> (if escalate_to_human is True, stress_score should always be >= 7),
        "primary_emotion": <one of: "calm","frustrated","anxious","angry","distressed","neutral">,
        "detected_language": <language name / mix or "Unknown" for noise or throat clear or blank>,
        "speech_content_summary": <one short sentence summarizing what the speaker said, or "Unable to determine">,
        "transcript": <best-effort VERBATIM transcript of what was said, in the ORIGINAL language/script spoken (use native script, e.g. Devanagari/Arabic/etc, not a romanization). Use "" if nothing intelligible could be transcribed>,
        "translated_text": <English translation of the transcript. If the speech was already in English, repeat it here. Use "" if there is no transcript to translate>,
        "emotion_breakdown": {"calm": <0-10>, "frustration": <0-10>, "anxiety": <0-10>, "anger": <0-10>},
        "distress_flag": <true/false — genuine personal/financial crisis signal, not just anger>,
        "distress_type": <null, or one of: "financial_hardship_crisis","health_emergency","severe_emotional_distress","threat_of_harm">,
        "recommended_agent_action": <one short, specific, de-escalating suggestion for a human agent>,
        "recommended_bot_script_branch": <one short label for how an AI voicebot script should adapt>, 
        "rationale": <one short sentence, plain language, no more than 25 words>,
        "in_depth_analysis": {
            "confidence": <float 0.0-1.0 — your own confidence in this entire reading>,
            "language_confidence": <float 0.0-1.0 — confidence in detected language and spoken-content understanding>,
            "content_vs_voice_weight": {
            "spoken_content": <0-100>,
            "vocal_characteristics": <0-100>
        },
        "key_evidence": [
            <3 to 6 short strings, each ONE concrete, specific acoustic cue or spoken phrase you actually heard and what it indicates — e.g.
            "pitch rose sharply on the word 'can't'",
            "said 'I lost my job last week'",
            "a 2-second pause before answering about the due date",
            "voice became shaky while saying 'please help'",
            "breathy, unsteady tone in the last third of the clip". Avoid vague statements like "sounded upset".>
        ],
        "stress_score_reasoning": <2-3 sentences: exactly why THIS numeric stress_score was chosen, explicitly referencing both the key evidence and spoken content where available>,
        "emotion_breakdown_reasoning": <1-2 sentences on why these relative emotion proportions were assigned rather than others>,
        "spoken_content_analysis": <2-4 sentences describing whether the spoken words indicate financial hardship, confusion, frustration, willingness to cooperate, threats, requests for help, or neutral conversation. If language cannot be reliably understood, explicitly state that>,
        "context_influence": <1-2 sentences: did the given account context — days past due, segment, prior complaint, call type — change how you weighted the vocal signal or spoken content, and how? Say "no material influence" if it didn't>,
        "alternative_explanations": <1-2 sentences: what ELSE, unrelated to genuine emotional distress, could produce this same vocal signature or wording — e.g. poor call quality, accent, naturally expressive speech, background noise, cultural speech variation>,
        "limitations": <1-2 sentences: what this single audio clip, judged in isolation, cannot actually tell you — e.g. no visibility into the customer's real financial situation, no baseline for this person's normal speaking voice, short clip length>
    },
    "acoustic_characteristics": [
        {"characteristic": <short name, e.g. "Sound clarity">, "score": <0-10>, "observation": <max 12 words>},
        ... 6 to 9 items covering DISTINCT vocal dimensions you can genuinely assess from
        this audio — draw from things like: sound/audio clarity, voice texture, pitch
        stability, speaking pace, volume consistency, breathiness or vocal strain,
        articulation, background noise level, vocal tremor, resonance/warmth, pause
        frequency — pick whichever ones the clip actually lets you judge, do not pad
        with items you cannot assess.
    ]
}
"""


def build_prompt(context: dict, features: dict = None) -> str:
    return f"""
You are an assistant supporting a bank's collections / customer-service call center.

Analyze the ATTACHED short audio clip of one side (or both sides) of a live phone
call for vocal stress, emotional state, and spoken content.

Your assessment MUST consider BOTH:

1. HOW the person speaks
    - pitch
    - speaking pace
    - pauses
    - vocal energy
    - breathiness
    - vocal strain
    - articulation
    - volume consistency
    - tremor
    - resonance
    - other observable acoustic characteristics

2. WHAT the person says
    - identify the spoken language
    - understand the meaning regardless of language
    - summarize the spoken content
    - detect mentions of financial hardship, illness, bereavement, job loss,
        emergencies, confusion, willingness to cooperate, payment intent,
        frustration, complaints, threats, or requests for help.

The speaker may use ANY language or mix multiple languages (code-switching).
Do NOT assume English.

If you understand the spoken language:
- identify it,
- transcribe what was said as accurately as possible in its ORIGINAL language and
  script (this is the "transcript" field),
- translate that transcript into English (the "translated_text" field) \u2014 skip
  translation only if it was already in English, in which case repeat it,
- summarize what was said,
- use BOTH spoken content and vocal delivery to determine the emotional state.

Transcription and translation are important here beyond just this one score \u2014
they are also used downstream for QA review, agent training, and future model
analysis, so prioritize faithfulness over polish. Partial or best-effort
transcripts are fine; do not fabricate words you are not reasonably confident
about, and do not refuse to attempt a transcript just because the audio is
imperfect.

If the language cannot be determined confidently:
- set detected_language to "unknown",
- state that spoken content could not be reliably interpreted,
- rely primarily on observable vocal characteristics,
- reduce confidence accordingly.

Never infer emotional distress solely from spoken words or solely from vocal tone.
Balance both sources of evidence whenever possible.

If spoken content and vocal delivery conflict (for example calm words spoken with
high vocal strain, or emotional words spoken calmly), explicitly explain this in
the reasoning and balance both forms of evidence.

Locally computed acoustic measurements for this exact clip (from signal processing,
NOT your own hearing — treat these as an objective, ground-truth reference to
cross-check what you hear, and call out explicitly in your reasoning if what you
hear seems to disagree with these numbers):
{format_features_for_prompt(features)}

Account context for this call (use it to calibrate severity, not to justify pressure):
- Days past due: {context.get('dpd')}
- Customer segment: {context.get('segment')}
- Prior complaint on file: {context.get('prior_complaint')}
- Call type: {context.get('call_type')}

Hard rule: this analysis exists to help the bank soften and de-escalate the
conversation and protect the customer — never to help extract payment from
someone in genuine distress. If you detect a genuine personal crisis, always
set escalate_to_human to true regardless of anything else in the call.

You are also being used to TRAIN and AUDIT — human agents, QA staff, and model
reviewers all read the "in_depth_analysis" section to understand exactly why the
model reached this conclusion, not just what the conclusion was. Be rigorous and
analytically honest there: cite specific, concrete things heard in the audio,
including both vocal cues and spoken phrases whenever possible, and reconcile
your read against the locally computed measurements above. State your own
confidence, note alternative explanations, and clearly describe what this single
clip cannot tell you. Do not simply restate the score in words.

{RESPONSE_SCHEMA_HINT}
""".strip()



def call_gemini(model_id: str, api_key: str, audio_bytes: bytes, mime_type: str, context: dict, features: dict = None):
    if not HAS_GENAI:
        raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)

    prompt = build_prompt(context, features)
    contents = [
        {"mime_type": mime_type or "audio/wav", "data": audio_bytes},
        prompt,
    ]

    generation_config = genai.GenerationConfig(
        temperature=0.25,
        response_mime_type="application/json",
    )

    response = model.generate_content(contents, generation_config=generation_config)
    raw_text = response.text or ""
    return parse_model_json(raw_text)

def parse_model_json(raw_text: str) -> dict:
    """Best-effort JSON extraction — model should already return clean JSON
    because of response_mime_type, but we defend against stray fences anyway."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    # Fallback safe default — never let a parse failure crash the app
    return {
        "stress_score": 0.0,
        "primary_emotion": "neutral",
        "detected_language": "unknown",
        "speech_content_summary": "Unable to determine.",
        "transcript": "",
        "translated_text": "",
        "emotion_breakdown": {
            "calm": 5,
            "frustration": 0,
            "anxiety": 0,
            "anger": 0,
        },
        "distress_flag": False,
        "distress_type": None,
        "recommended_agent_action": "Model response could not be parsed — treat as inconclusive.",
        "recommended_bot_script_branch": "default",
        "escalate_to_human": False,
        "rationale": "Parsing failed; showing safe default.",
        "in_depth_analysis": {
            "confidence": 0.0,
            "language_confidence": 0.0,
            "content_vs_voice_weight": {
                "spoken_content": 0,
                "vocal_characteristics": 100,
            },
            "key_evidence": [],
            "stress_score_reasoning": (
                "The model's response could not be parsed into structured data, "
                "so no reasoning is available for this turn."
            ),
            "emotion_breakdown_reasoning": "Not available — parse failure.",
            "spoken_content_analysis": (
                "Spoken language and content could not be analyzed because the "
                "model response could not be parsed."
            ),
            "context_influence": "Not available — parse failure.",
            "alternative_explanations": "Not available — parse failure.",
            "limitations": (
                "This entire turn's output is a safe default, not a real model "
                "analysis. Re-run the analysis."
            ),
        },
        "acoustic_characteristics": [],
        "_parse_error": True,
        "_raw_text": raw_text[:500],
    }
    

def enforce_hard_guardrails(result: dict) -> dict:
    """Non-negotiable escalation floor — code-level, cannot be overridden by the model."""
    crisis_types = {"financial_hardship_crisis", "health_emergency", "severe_emotional_distress", "threat_of_harm"}
    if result.get("distress_flag") or result.get("distress_type") in crisis_types or result.get("stress_score", 0) >= 7:
        result["escalate_to_human"] = True
    return result


# =================================================================================
# 6. VISUAL HELPERS
# =================================================================================

def risk_color(score: float) -> str:
    if score >= 7:
        return RED
    if score >= 4:
        return AMBER
    return GREEN


def stress_gauge(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 10", "font": {"size": 34, "color": INK}},
            gauge={
                "axis": {"range": [0, 10], "tickcolor": MUTED},
                "bar": {"color": risk_color(score)},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 4], "color": "#E7F5EC"},
                    {"range": [4, 7], "color": "#FBF0DD"},
                    {"range": [7, 10], "color": "#FBE4E4"},
                ],
                "threshold": {
                    "line": {"color": RED, "width": 3},
                    "thickness": 0.85,
                    "value": 7,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def emotion_radar(breakdown: dict):
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=[l.capitalize() for l in labels_closed],
            fill="toself",
            line_color=TEAL,
            fillcolor="rgba(28,114,147,0.35)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=280,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def waveform_chart(samples, sr):
    if samples is None:
        return None
    max_points = 3000
    step = max(1, len(samples) // max_points)
    x = np.arange(0, len(samples), step) / sr
    y = samples[::step]
    fig = go.Figure(go.Scatter(x=x, y=y, line=dict(color=TEAL, width=1)))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="seconds",
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

    
def timeline_chart(history):
    if not history:
        return None

    turns = list(range(1, len(history) + 1))
    scores = [h["result"]["stress_score"] for h in history]
    timestamps = [h["timestamp"] for h in history]
    colors = [risk_color(s) for s in scores]

    fig = go.Figure(
        go.Scatter(
            x=timestamps,
            y=scores,
            mode="lines+markers",
            line=dict(color=TEAL, width=2),
            marker=dict(size=10, color=colors),
            customdata=turns,
            hovertemplate=(
                "<b>Turn %{customdata}</b><br>"
                "Time: %{x}<br>"
                "Stress score: %{y}<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=7,
        line_dash="dash",
        line_color=RED,
        annotation_text="Escalation floor"
    )

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Timestamp",
        yaxis=dict(
            title="Stress score",
            range=[0, 10]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =================================================================================
# 7. SIDEBAR — controls
# =================================================================================

with st.sidebar:
    choice = st.checkbox("Offline Model", value=False, key="offline_model_check")
    # st.markdown("### \U0001F511 Gemini API")
    # api_key_input = st.text_input(
    #     "API key", value=st.session_state.api_key, type="password",
    #     help="Get a free key at https://aistudio.google.com/apikey — kept only in this session, never saved to disk.",
    # )
    # st.session_state.api_key = api_key_input
    if choice:
        st.session_state.api_key = ""
    else:
        st.session_state.api_key = st.secrets(["GEMINI_API_KEY"])

    st.markdown("### \U0001F9E0 Model")
    model_id = st.selectbox(
        "Choose an audio-capable Gemini model",
        options=list(MODEL_CATALOG.keys()),
        format_func=lambda k: MODEL_CATALOG[k]["label"],
        index=list(MODEL_CATALOG.keys()).index(DEFAULT_MODEL),
    )
    st.caption(MODEL_CATALOG[model_id]["notes"])
    st.caption(f"Speed tier: **{MODEL_CATALOG[model_id]['speed_tier']}**")

    st.markdown("---")
    st.markdown("### \U0001F4CB Account context")
    dpd = st.slider("Days past due", 0, 180, 15)
    segment = st.selectbox("Customer segment", ["Retail", "MSME", "Priority sector", "HNI"])
    prior_complaint = st.checkbox("Prior complaint on file")
    call_type = st.selectbox("Call type", ["Collections / recovery", "Customer service", "Retention"], index=1)

    st.markdown("---")
    if st.button("\U0001F5D1\uFE0F Clear session history"):
        st.session_state.history = []
        st.rerun()

    if not st.session_state.api_key:
        st.info("No API key yet \u2014 the app will run in **offline heuristic mode** (local acoustic scoring only, no LLM reasoning).")


# =================================================================================
# 8. MAIN — hero + recorder
# =================================================================================

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="kicker">BFSI \u00b7 DS/ML POC</div>
        <div class="hero-title">\U0001F3A7 Emotion-Aware Call Monitor</div>
        <div class="hero-sub">Record a call snippet \u2192 real-time voice-stress read \u2192 live agent guidance, before the call ends.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

rec_col, ctx_col = st.columns([2, 1])

with rec_col:
    st.markdown("#### \U0001F399\uFE0F Record a call snippet")
    audio_value = st.audio_input("Click to record (speak for 3-10 seconds, then stop)")

with ctx_col:
    st.markdown("#### \u2139\uFE0F This turn's context")
    st.markdown(
        f"""
        <div class="poc-card">
        <b>DPD:</b> {dpd} days<br>
        <b>Segment:</b> {segment}<br>
        <b>Prior complaint:</b> {"Yes" if prior_complaint else "No"}<br>
        <b>Call type:</b> {call_type}
        </div>
        """,
        unsafe_allow_html=True,
    )

analyze_clicked = st.button("\u26A1 Analyze this turn", type="primary", disabled=(audio_value is None))


# =================================================================================
# 9. ANALYSIS RUN
# =================================================================================

if analyze_clicked and audio_value is not None:
    audio_bytes = audio_value.getvalue()
    mime_type = getattr(audio_value, "type", None) or "audio/wav"

    with st.spinner("Extracting voice features..."):
        samples, sr, features = extract_prosody_features(audio_bytes)

    context = {"dpd": dpd, "segment": segment, "prior_complaint": prior_complaint, "call_type": call_type}

    result = None
    mode_used = "offline"
    error_msg = None

    if st.session_state.api_key:
        try:
            with st.spinner(f"Sending audio to {MODEL_CATALOG[model_id]['label']}..."):
                t0 = time.time()
                result = call_gemini(model_id, st.session_state.api_key, audio_bytes, mime_type, context, features)
                latency = round(time.time() - t0, 2)
            mode_used = "gemini"
        except Exception as e:
            error_msg = str(e)

    if result is None:
        # offline heuristic fallback
        score = heuristic_stress_score(features)
        result = {
            "stress_score": score,
            "primary_emotion": "frustrated" if score >= 5 else "calm",
            "detected_language": "Not available (offline mode has no speech-to-text)",
            "speech_content_summary": "Not available \u2014 offline mode only analyzes acoustic signal, not spoken content.",
            "transcript": "",
            "translated_text": "",
            "emotion_breakdown": {
                "calm": max(0, 10 - score),
                "frustration": min(score, 7),
                "anxiety": min(score * 0.6, 6),
                "anger": min(score * 0.4, 5),
            },
            "distress_flag": False,
            "distress_type": None,
            "recommended_agent_action": "Offline mode: acoustic signal only — add an API key for contextual guidance.",
            "recommended_bot_script_branch": "default",
            "escalate_to_human": score >= 7,
            "rationale": "Estimated from pitch variance, energy and pace only (no LLM reasoning).",
            "in_depth_analysis": offline_in_depth_analysis(features, score),
            "acoustic_characteristics": offline_acoustic_table(features),
        }
        latency = None

    result = enforce_hard_guardrails(result)

    st.session_state.history.append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "result": result,
            "features": features,
            "mode": mode_used,
            "model": model_id if mode_used == "gemini" else "offline-heuristic",
            "latency": latency,
        }
    )

    if error_msg:
        st.error(f"Gemini call failed, showing offline estimate instead. Details: {error_msg}")

    # ---- Escalation / OK banner ----
    if result["escalate_to_human"]:
        st.markdown(
            f'<div class="escalate-banner">\U0001F6A8 ESCALATE TO HUMAN \u2014 {result["rationale"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ok-banner">\u2705 No escalation needed this turn \u2014 {result["rationale"]}</div>',
            unsafe_allow_html=True,
        )

    # ---- Main results row ----
    g1, g2, g3 = st.columns([1, 2, 1.2])

    with g1:
        st.markdown("**Stress score**")
        st.plotly_chart(stress_gauge(result["stress_score"]), use_container_width=True, config={"displayModeBar": False})

    with g2:
        st.markdown("**Emotion breakdown**")
        g21, g22 = st.columns([1, 1])
        with g21:
            st.plotly_chart(emotion_radar(result["emotion_breakdown"]), use_container_width=True, config={"displayModeBar": False})
        with g22:
            emotion_df = pd.DataFrame([result["emotion_breakdown"]])
            emotion_df = emotion_df.transpose().reset_index()

            emotion_df.columns = ["Emotion", "value"]
            
            total = emotion_df["value"].sum()

            if total > 0:
                emotion_df["value"] = emotion_df["value"] / total * 10

                # Round
                emotion_df["value"] = emotion_df["value"].round(2)

                # Fix rounding difference
                diff = 10 - emotion_df["value"].sum()
                emotion_df.loc[emotion_df["value"].idxmax(), "value"] += diff

            emotion_df = emotion_df.sort_values(
                by="value",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                emotion_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Emotion": st.column_config.TextColumn(
                        "Emotion"
                    ),
                    "value": st.column_config.ProgressColumn(
                        "Score (0-10)",
                        min_value=0,
                        max_value=10,
                        format="%.2f"
                    )
                },
            )
        

    with g3:
        st.markdown("**Guidance**")
        emo = result["primary_emotion"].capitalize()
        badge_color = risk_color(result["stress_score"])

        if result.get("_parse_error"):
            source_label = "model response unparsed \u2014 safe default shown"
        elif mode_used == "gemini":
            source_label = f"Gemini: {model_id}"
        else:
            source_label = "offline heuristic (no API key)"

        st.markdown(
            f"""
            <div class="poc-card">
            <span class="badge" style="background:{badge_color};">{emo}</span>
            &nbsp;&nbsp;<span style="color:{MUTED}; font-size:0.85rem;">via {source_label}</span>
            <hr style="margin:0.6rem 0;">
            <b>Agent action:</b> {result['recommended_agent_action']}<br><br>
            <b>Bot script branch:</b> {result['recommended_bot_script_branch']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if latency:
            st.caption(f"Model latency: {latency}s")

    # ---- Waveform ----
    wf = waveform_chart(samples, sr) if samples is not None else None
    if wf is not None:
        st.markdown("**Waveform**")
        st.plotly_chart(wf, use_container_width=True, config={"displayModeBar": False})
    elif not (HAS_SOUNDFILE and HAS_LIBROSA):
        st.caption("Install `soundfile` and `librosa` to see the waveform panel.")

    # ---- Language & Transcript ----
    st.markdown("**\U0001F310 Language & transcript**")
    lang = result.get("detected_language") or "Unknown"
    lang_conf = (result.get("in_depth_analysis") or {}).get("language_confidence")
    transcript = result.get("transcript") or ""
    translated = result.get("translated_text") or ""
    summary = result.get("speech_content_summary") or ""

    lcol1, lcol2 = st.columns([1, 2])
    with lcol1:
        st.markdown(
            f"""
            <div class="poc-card">
            <b>Detected language:</b> {lang}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if lang_conf is not None:
            st.caption(f"Language confidence: {lang_conf * 100:.0f}%")
            st.progress(min(max(lang_conf, 0.0), 1.0))
    with lcol2:
        st.markdown(
            f"""
            <div class="poc-card">
            <b>Transcript (original language):</b><br>{transcript if transcript else "<i>Not available</i>"}
            <hr style="margin:0.6rem 0;">
            <b>Translation (English):</b><br>{translated if translated else "<i>Not available</i>"}
            </div>
            """,
            unsafe_allow_html=True,
        )
    if summary:
        st.caption(f"Summary: {summary}")

    # ---- In-depth analysis (for QA / model audit) ----
    ida = result.get("in_depth_analysis") or {}
    if ida:
        with st.expander("\U0001F52C In-depth analysis (for QA / model audit)", expanded=True):
            conf = ida.get("confidence")
            if conf is not None:
                st.markdown(f"**Model's own confidence in this read:** {conf * 100:.0f}%")
                st.progress(min(max(conf, 0.0), 1.0))

            evidence = ida.get("key_evidence") or []
            if evidence:
                st.markdown("**Key acoustic evidence**")
                for point in evidence:
                    st.markdown(f"- {point}")

            st.markdown("**Why this stress score**")
            st.markdown(ida.get("stress_score_reasoning", "\u2014"))

            st.markdown("**Why this emotion breakdown**")
            st.markdown(ida.get("emotion_breakdown_reasoning", "\u2014"))

            st.markdown("**What the spoken content indicates**")
            st.markdown(ida.get("spoken_content_analysis", "\u2014"))

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**How account context influenced this read**")
                st.markdown(ida.get("context_influence", "\u2014"))
            with colB:
                st.markdown("**Alternative explanations (not necessarily distress)**")
                st.markdown(ida.get("alternative_explanations", "\u2014"))

            st.markdown("**Limitations of this reading**")
            st.markdown(ida.get("limitations", "\u2014"))

            st.caption("Shown in full for internal analysis, agent training, and model audit \u2014 not just the headline score.")

    # ---- Acoustic characteristic scorecard ----
    acoustic_rows = result.get("acoustic_characteristics") or []
    if acoustic_rows:
        st.markdown("**Acoustic characteristic scorecard**")
        if mode_used != "gemini":
            st.caption("Offline proxy scores (signal-processing only) — connect a Gemini model for a richer, model-judged read.")

        df = pd.DataFrame(acoustic_rows)
        df = df.rename(columns={
            "characteristic": "Characteristic",
            "score": "Score",
            "observation": "Observation",
        })
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score (0-10)", min_value=0, max_value=10, format="%.1f"
                ),
                "Characteristic": st.column_config.TextColumn("Characteristic", width="medium"),
                "Observation": st.column_config.TextColumn("Observation", width="large"),
            },
        )


# =================================================================================
# 10. SESSION TIMELINE
# =================================================================================

st.markdown("---")
st.markdown("#### \U0001F4C8 Session timeline")

if st.session_state.history:
    tl = timeline_chart(st.session_state.history)
    st.plotly_chart(tl, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Raw turn-by-turn log"):
        for i, turn in enumerate(reversed(st.session_state.history), start=1):
            r = turn["result"]
            st.write(
                f"**Turn {len(st.session_state.history) - i + 1}** \u2014 {turn['timestamp']} \u2014 "
                f"model: `{turn['model']}` \u2014 score: **{r['stress_score']}** \u2014 "
                f"escalate: **{r['escalate_to_human']}**"
            )
else:
    st.caption("No turns analyzed yet in this session.")


# =================================================================================
# 11. COMPLIANCE FOOTER
# =================================================================================

st.markdown(
    f"""
    <div class="footer-note">
    \u26A0\uFE0F <b>POC disclaimer:</b> for demonstration only. Not connected to a production
    telephony system. Designed to <i>protect</i> the customer \u2014 escalation logic is a hard
    floor enforced in code and cannot be overridden by model output. Voice-based emotion
    analysis requires its own explicit consent disclosure in production, separate from
    standard call-recording consent.
    </div>
    """,
    unsafe_allow_html=True,
)
