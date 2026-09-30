# 🎙️ Emotion-Aware Collections & Service Calls

### Bank AI / BFSI — Streamlit Proof of Concept

An **AI-powered voice analytics proof of concept** for collections and customer-service calls that combines:

* 🎙️ Browser-based audio recording
* 🔊 Local acoustic / prosodic feature extraction
* 🤖 Gemini native-audio analysis
* 😟 Stress and emotion detection
* 🌐 Multilingual speech understanding
* 📝 Original-language transcription
* 🔤 English translation
* 🧠 Context-aware reasoning
* 🚨 Hard-coded human escalation guardrails
* 👨‍💼 Real-time agent guidance
* 📊 Acoustic characteristic scoring
* 📈 Multi-turn session stress timeline
* 📴 Offline heuristic fallback

The POC is designed around a **customer-protection and de-escalation workflow**, rather than using emotion analysis to pressure customers into making payments.

---

## 📌 Overview

The application records a short **3–10 second call snippet** through the browser and analyzes the speaker's vocal characteristics and, when Gemini is enabled, the spoken content.

The system evaluates two complementary evidence sources:

### 1. How the customer speaks

The application considers acoustic characteristics such as:

* Pitch
* Pitch variability
* Speaking pace
* Pauses
* Vocal energy
* Volume consistency
* Breathiness
* Vocal strain
* Articulation
* Tremor
* Voice texture
* Zero-crossing rate
* Background/noise characteristics

### 2. What the customer says

When Gemini is available, the system also analyzes:

* Spoken language
* Code-switching / multilingual speech
* Transcript
* English translation
* Financial hardship
* Health emergencies
* Job loss
* Bereavement
* Confusion
* Payment intent
* Complaints
* Requests for help
* Threats
* Willingness to cooperate

These signals are combined to produce a structured analysis and recommended agent response.

---

# 🏦 Business Use Case

The POC is intended for **Banking, Financial Services and Insurance (BFSI)** environments where customer-service or collections agents may benefit from real-time assistance.

### Example workflow

```text
Customer speaks
       │
       ▼
Browser audio recording
       │
       ▼
Local acoustic analysis
       │
       ├───────────────┐
       │               │
       ▼               ▼
Prosodic features   Gemini audio analysis
       │               │
       │        Speech + emotion
       │        + language + context
       │               │
       └───────┬───────┘
               ▼
       Structured analysis
               │
       ├───────────────┬────────────────┐
       ▼               ▼                ▼
  Stress score    Agent guidance   Distress detection
       │               │                │
       └───────────────┴────────────────┘
                       ▼
              Human escalation
              when required
```

---

# ✨ Key Features

## 🎙️ Browser-Based Audio Recording

The application uses Streamlit's audio input component:

```python
st.audio_input(...)
```

The user can record a short call snippet directly from the browser.

Recommended recording duration:

```text
3–10 seconds
```

---

## 🔊 Local Acoustic Feature Extraction

When `soundfile` and `librosa` are available, the application performs local signal processing.

Extracted features include:

| Feature            | Description                    |
| ------------------ | ------------------------------ |
| Duration           | Length of the audio clip       |
| Mean pitch         | Average fundamental frequency  |
| Pitch variability  | Standard deviation of F0       |
| Mean energy        | Average RMS energy             |
| Energy variability | Variation in vocal energy      |
| Zero-crossing rate | Signal texture/noisiness proxy |

The extraction is performed locally and is independent of the Gemini API.

---

# 🧠 Gemini Audio Analysis

When a Gemini API key is available, the audio is sent directly to an audio-capable Gemini model.

The model evaluates both:

```text
Vocal characteristics
+
Spoken content
+
Account context
```

The application explicitly instructs the model **not to infer emotional distress from vocal tone or words alone**.

Instead, it should balance both evidence sources whenever possible.

---

# 🌐 Multilingual Speech Understanding

The Gemini analysis supports speech that may contain:

* English
* Indian languages
* Other international languages
* Mixed-language conversations
* Code-switching

The response attempts to provide:

### Detected language

Example:

```text
Hindi
```

or:

```text
Hindi + English
```

### Original transcript

The transcript is requested in the **original language and script**.

For example:

```text
मुझे इस महीने भुगतान करने में बहुत कठिनाई हो रही है।
```

### English translation

```text
I am having a lot of difficulty making the payment this month.
```

The application also displays a short spoken-content summary.

---

# 😟 Stress Analysis

The system produces a stress score between:

```text
0 ─────────────── 10
Calm             Extreme distress
```

The application uses three broad visual ranges:

|   Score | Interpretation                     |
| ------: | ---------------------------------- |
| `0–3.9` | Lower stress                       |
| `4–6.9` | Elevated stress                    |
|  `7–10` | High stress / escalation threshold |

The score is displayed through an interactive Plotly gauge.

---

# 🚨 Human Escalation Guardrail

One of the most important features of the POC is that escalation is **not controlled exclusively by the LLM**.

The application implements a code-level hard guardrail:

```python
if (
    result.get("distress_flag")
    or result.get("distress_type") in crisis_types
    or result.get("stress_score", 0) >= 7
):
    result["escalate_to_human"] = True
```

The crisis categories include:

* `financial_hardship_crisis`
* `health_emergency`
* `severe_emotional_distress`
* `threat_of_harm`

Additionally:

```text
stress_score >= 7
```

automatically triggers human escalation.

### Important principle

The model cannot override the escalation floor.

```text
Model output
     │
     ▼
Code-level guardrail
     │
     ├── Crisis detected ──► Human
     │
     ├── Stress >= 7 ──────► Human
     │
     └── Otherwise ────────► Continue
```

---

# 👨‍💼 Agent Guidance

The application generates a short recommended action for the human agent.

Example output structure:

```text
Agent action:
Acknowledge the customer's concern and offer to review available assistance options.

Bot script branch:
empathetic_assistance
```

The system is designed to help the agent **de-escalate and support the customer**.

---

# 🤖 AI Voicebot Script Branch

The model also returns:

```text
recommended_bot_script_branch
```

This can be used downstream to dynamically select an appropriate conversational branch.

For example:

```text
default
empathetic_assistance
clarification
human_escalation
financial_hardship
```

The exact labels are model-generated and can be mapped to a production conversation engine.

---

# 🧠 In-Depth AI Analysis

The application exposes an expandable:

> 🔬 In-depth analysis (for QA / model audit)

section.

This is intended for:

* QA teams
* Model reviewers
* Agent training
* AI governance
* POC validation
* Error analysis

The analysis contains:

### Confidence

The model's confidence in the overall interpretation.

### Key evidence

Concrete acoustic or spoken evidence, for example:

```text
Pitch rose sharply while discussing the payment.
```

or:

```text
Customer stated that they lost their job last week.
```

### Stress-score reasoning

Explains why the particular numerical score was selected.

### Emotion-breakdown reasoning

Explains the relative distribution of:

* Calm
* Frustration
* Anxiety
* Anger

### Spoken-content analysis

Evaluates whether the spoken content indicates:

* Financial hardship
* Confusion
* Frustration
* Requests for help
* Payment intent
* Threats
* Neutral conversation

### Context influence

Explains whether account context affected the interpretation.

### Alternative explanations

The model is explicitly asked to consider non-distress explanations such as:

* Background noise
* Poor call quality
* Accent
* Naturally expressive speech
* Cultural speech variation
* Individual speaking style

### Limitations

The model is asked to explain what cannot be established from a single short audio clip.

---

# 📊 Emotion Breakdown

The application returns an emotion breakdown containing:

```text
Calm
Frustration
Anxiety
Anger
```

The results are displayed using:

* Radar visualization
* Normalized score table
* Progress bars

The radar chart provides a visual representation of the relative emotional signals detected in the turn.

---

# 🎚️ Acoustic Characteristic Scorecard

Gemini can return multiple acoustic characteristics that can be assessed from the recording.

Possible dimensions include:

* Sound clarity
* Voice texture
* Pitch stability
* Speaking pace
* Volume consistency
* Breathiness
* Vocal strain
* Articulation
* Background noise
* Vocal tremor
* Resonance
* Pause frequency

Each characteristic contains:

```text
Characteristic
Score: 0–10
Observation
```

Example:

| Characteristic     | Score | Observation                 |
| ------------------ | ----: | --------------------------- |
| Pitch stability    |   6.5 | Moderate pitch variation    |
| Volume consistency |   5.8 | Noticeable volume changes   |
| Speaking pace      |   4.9 | Faster-than-normal delivery |
| Voice texture      |   6.1 | Slight vocal strain         |

The exact characteristics depend on what the model can genuinely assess from the recording.

---

# 📈 Session Timeline

Every analyzed turn is stored in:

```python
st.session_state.history
```

The application builds a session-level timeline showing:

* Timestamp
* Stress score
* Turn number
* Escalation threshold
* Model used

The chart includes a reference line at:

```text
Stress = 7
```

This allows the user to observe whether stress is:

```text
Increasing
     ↓
Stable
     ↓
Decreasing
```

throughout the current session.

---

# 📴 Offline Mode

The application can run without a Gemini API key.

Enable:

```text
Offline Model
```

in the sidebar.

Offline mode uses locally calculated acoustic features and a simple heuristic stress score.

The heuristic considers:

* Pitch variability
* Energy variability
* Zero-crossing rate

Conceptually:

```text
Pitch variability
       +
Energy variability
       +
Zero-crossing rate
       ↓
Heuristic stress score
```

### Important

Offline mode **does not provide true speech understanding**.

It cannot reliably provide:

* Speech-to-text
* Language detection
* Translation
* Semantic understanding
* Contextual reasoning
* Reliable emotion classification

Therefore, offline results are explicitly presented as:

```text
offline heuristic
```

rather than an LLM-based assessment.

---

# 🧮 Offline Stress Heuristic

The fallback calculates a bounded score from acoustic characteristics.

Conceptually:

```python
score =
    pitch_variability_component
    + energy_variability_component
    + zero_crossing_component
```

The final value is constrained to:

```text
0–10
```

This is intentionally simple and transparent.

It is **not a trained emotion-recognition model**.

---

# 🛡️ Safe Fallback on Gemini Failure

If the Gemini API call fails, the application does not crash.

Instead:

```text
Gemini request
      │
      ├── Success ──► Gemini analysis
      │
      └── Failure ──► Offline heuristic
```

The UI displays an error message and continues with the locally generated estimate.

This makes the POC more resilient during:

* API failures
* Invalid model IDs
* Network errors
* Quota issues
* Temporary service problems

---

# 🧾 Structured JSON Output

Gemini is instructed to return a structured JSON response.

The primary schema contains:

```text
escalate_to_human
stress_score
primary_emotion
detected_language
speech_content_summary
transcript
translated_text
emotion_breakdown
distress_flag
distress_type
recommended_agent_action
recommended_bot_script_branch
rationale
in_depth_analysis
acoustic_characteristics
```

This structure makes the output suitable for downstream integration with:

* Agent-assist applications
* QA systems
* CRM platforms
* Contact-center platforms
* Analytics dashboards
* Conversation intelligence pipelines

---

# 🔐 Account Context

The application allows contextual information to be supplied for the current call:

### Days Past Due

```text
0–180 days
```

### Customer Segment

Available options:

```text
Retail
MSME
Priority sector
HNI
```

### Prior Complaint

```text
Yes / No
```

### Call Type

```text
Collections / recovery
Customer service
Retention
```

The context is used to **calibrate interpretation**, not to justify pressure on the customer.

The prompt explicitly instructs the model to use context responsibly.

---

# 🧩 Model Catalog

The application maintains a configurable model catalog:

```python
MODEL_CATALOG = {
    ...
}
```

The catalog is designed to contain audio-capable Gemini models.

The selected model is shown in the sidebar together with:

* Model description
* Speed tier
* Intended usage

Example speed categories:

```text
fastest
fast
deep
```

These are application-level labels for expected latency characteristics, not formal API performance specifications.

---

# 🔄 Model Maintenance

Generative AI model identifiers can change over time.

If a configured model returns:

```text
404
model not found
```

update the model identifier in:

```python
MODEL_CATALOG
```

The rest of the application architecture does not need to change.

Always verify currently supported Gemini models in Google's documentation before deploying a model change.

---

# 🎨 User Interface

The application uses a professional BFSI-oriented visual theme.

Primary colors include:

| Color     | Purpose                              |
| --------- | ------------------------------------ |
| Navy      | Main application identity            |
| Deep Navy | Banner/background                    |
| Teal      | Charts and analytical elements       |
| Amber     | Attention / informational highlights |
| Green     | Normal / no escalation               |
| Red       | Escalation                           |
| Off-white | Application background               |

The interface contains:

```text
Hero banner
      ↓
Audio recorder
      ↓
Current call context
      ↓
Analyze button
      ↓
Escalation status
      ↓
Stress gauge
      ↓
Emotion analysis
      ↓
Agent guidance
      ↓
Waveform
      ↓
Language & transcript
      ↓
In-depth QA analysis
      ↓
Acoustic scorecard
      ↓
Session timeline
```

---

# 🏗️ Application Architecture

```text
emotion_aware_calls_poc.py
│
├── Streamlit UI
│
├── Audio Input
│
├── Local Signal Processing
│   ├── soundfile
│   └── librosa
│
├── Prosodic Feature Extraction
│   ├── RMS energy
│   ├── Zero crossing rate
│   └── Pitch / F0
│
├── Gemini Audio Analysis
│   └── google-generativeai
│
├── JSON Parser
│
├── Hard Escalation Guardrails
│
├── Visualization
│   ├── Stress Gauge
│   ├── Emotion Radar
│   ├── Waveform
│   └── Session Timeline
│
└── Session History
```

---

# 🛠️ Technology Stack

| Technology               | Purpose                      |
| ------------------------ | ---------------------------- |
| Python                   | Application runtime          |
| Streamlit                | Web application              |
| Gemini                   | Audio + language + reasoning |
| Google Generative AI SDK | Gemini API integration       |
| Librosa                  | Audio analysis               |
| SoundFile                | Audio loading                |
| NumPy                    | Numerical processing         |
| Pandas                   | Tabular analysis             |
| Plotly                   | Interactive visualization    |
| Regex / JSON             | Structured response parsing  |

---

# 📦 Requirements

A typical `requirements.txt` can contain:

```text
streamlit
numpy
pandas
plotly
soundfile
librosa
google-generativeai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you only need the basic application without local acoustic analysis, `soundfile` and `librosa` are optional because the application handles their absence gracefully.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Gemini API

Create a Gemini API key through Google's AI Studio.

Store the key as a Streamlit secret.

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do **not** commit this file to GitHub.

---

# ▶️ Run the Application

```bash
streamlit run emotion_aware_calls_poc.py
```

The application will open in the browser.

---

# 🧪 How to Use

## Step 1 — Select the operating mode

From the sidebar choose:

```text
Offline Model
```

or Gemini-based analysis.

---

## Step 2 — Select Gemini Model

Choose an audio-capable model from the model dropdown.

The default configured model is:

```text
gemini-flash-lite-latest
```

---

## Step 3 — Configure Call Context

Set:

```text
Days past due
Customer segment
Prior complaint
Call type
```

---

## Step 4 — Record Audio

Click:

```text
🎙️ Record a call snippet
```

Speak for approximately:

```text
3–10 seconds
```

and stop the recording.

---

## Step 5 — Analyze

Click:

```text
⚡ Analyze this turn
```

The application performs:

```text
Audio extraction
      ↓
Local acoustic analysis
      ↓
Gemini analysis
      ↓
JSON parsing
      ↓
Guardrail enforcement
      ↓
Visualization
```

---

## Step 6 — Review Results

Review:

* Stress score
* Primary emotion
* Emotion breakdown
* Escalation status
* Agent guidance
* Bot script branch
* Waveform
* Detected language
* Original transcript
* English translation
* In-depth reasoning
* Acoustic characteristics

---

# 📋 Example Output

A Gemini analysis may conceptually produce:

```json
{
  "escalate_to_human": true,
  "stress_score": 7.4,
  "primary_emotion": "distressed",
  "detected_language": "Hindi",
  "speech_content_summary": "The customer describes difficulty making the current payment.",
  "transcript": "मुझे इस महीने भुगतान करने में बहुत कठिनाई हो रही है।",
  "translated_text": "I am having a lot of difficulty making the payment this month.",
  "distress_flag": true,
  "distress_type": "financial_hardship_crisis",
  "recommended_agent_action": "Acknowledge the hardship and move to an appropriate assistance pathway.",
  "recommended_bot_script_branch": "financial_hardship"
}
```

The exact output depends on the recorded audio and selected model.

---

# 🔬 Why Both Voice and Speech Matter

A major design principle of this POC is that **emotion cannot reliably be determined from a single signal source**.

For example:

### Scenario A

```text
Words: "Everything is fine."
Voice: shaky, strained and highly variable
```

The system should recognize that the verbal and acoustic signals differ and explain the conflict.

### Scenario B

```text
Words: complaint
Voice: calm and cooperative
```

The system should not automatically interpret the complaint as severe emotional distress.

The prompt therefore instructs Gemini to explicitly reconcile conflicting evidence.

---

# 🧠 Explainability Design

The POC is designed to expose more than a single score.

Instead of:

```text
Stress = 8.2
```

the system attempts to provide:

```text
Stress = 8.2

Why?
 ├── Pitch variation
 ├── Vocal strain
 ├── Speaking pattern
 ├── Specific spoken phrase
 ├── Context
 └── Alternative explanations
```

This supports:

* Human review
* Model auditing
* QA analysis
* Agent training
* POC validation

---

# ⚠️ Limitations

This is a **proof of concept**, not a production emotion-detection system.

Important limitations include:

### Short audio duration

A 3–10 second clip provides limited evidence.

### Individual speaking differences

People naturally differ in:

* Pitch
* Loudness
* Speaking speed
* Accent
* Voice texture

### Audio quality

Background noise, compression and poor microphones can influence acoustic measurements.

### No individual baseline

The application does not maintain a reliable baseline of each customer's normal voice.

### Emotion is difficult to infer

Vocal characteristics do not provide a definitive measurement of a person's emotional or psychological state.

### Offline mode is especially limited

Offline mode uses signal-processing heuristics and has no semantic understanding.

### Context can bias interpretation

Account information should be used carefully and should not be treated as proof of distress.

---

# 🔐 Privacy & Security Considerations

In a production implementation, audio handling should be designed according to the bank's applicable privacy, security, retention and consent requirements.

Important considerations include:

* Explicit consent for voice recording
* Separate disclosure for voice/emotion analysis where required
* Encryption in transit
* Encryption at rest
* Restricted API access
* Appropriate audio retention periods
* Access controls
* Audit logging
* PII handling
* Data residency requirements
* Vendor/API governance
* Model governance
* Human oversight

### API key

Never hard-code:

```python
GEMINI_API_KEY = "..."
```

in the source code.

Use Streamlit Secrets or an appropriate enterprise secret-management mechanism.

---

# 🚨 Production Consent Requirement

The application footer explicitly highlights that production deployment requires an appropriate consent/disclosure mechanism.

Voice recording consent and emotion-analysis disclosure may have different requirements.

Therefore, a production contact-center implementation should work with the relevant:

* Legal
* Compliance
* Information Security
* Privacy
* Risk
* Customer-protection

teams before deployment.

---

# 🏦 Production Architecture — Possible Extension

A production architecture could evolve toward:

```text
Telephony / Contact Center
          │
          ▼
     Audio Stream
          │
          ▼
   Secure Audio Gateway
          │
     ┌────┴────┐
     ▼         ▼
 Acoustic    Gemini /
 Analysis    AI Layer
     │         │
     └────┬────┘
          ▼
   Emotion + Content
       Analysis
          │
          ▼
   Guardrail Engine
          │
    ┌─────┴─────┐
    ▼           ▼
Agent Assist   Escalation
    │           │
    ▼           ▼
 CRM / CCaaS   Human Agent
```

Additional production components could include:

* Real-time streaming
* Call transcription
* CRM integration
* Agent desktop integration
* Contact-center integration
* Model monitoring
* Human review
* Audit trails
* Consent management
* Data retention controls

---

# 🔮 Future Enhancements

Potential future development areas include:

### Real-Time Streaming

Move from short recorded snippets to streaming call analysis.

### Agent Desktop Integration

Display recommendations directly inside the agent's CRM/contact-center interface.

### Customer Baseline

Build a privacy-controlled baseline of the customer's normal speech characteristics where appropriate and permitted.

### Advanced Acoustic Models

Introduce dedicated speech/emotion models alongside Gemini.

### Speaker Diarization

For two-sided calls:

```text
Customer
Agent
```

could be separated automatically.

### Conversation-Level Analysis

Aggregate multiple turns instead of analyzing isolated clips.

### Escalation Workflows

Automatically route qualifying calls to:

```text
Human specialist
Customer support
Hardship team
Emergency workflow
```

according to approved business rules.

### QA Dashboard

Track:

* Stress trends
* Escalation rates
* Model confidence
* Language distribution
* Call categories
* Agent response patterns

### Model Evaluation

Build a labeled validation dataset and evaluate:

* Precision
* Recall
* F1
* False escalation rate
* Missed escalation rate
* Language-specific performance
* Audio-quality sensitivity

---

# 📊 Suggested Evaluation Framework

Before production use, evaluate the system separately for:

| Dimension             | Example metric                         |
| --------------------- | -------------------------------------- |
| Stress classification | Precision / Recall / F1                |
| Escalation            | False positive / false negative rate   |
| Language detection    | Accuracy                               |
| Transcription         | WER                                    |
| Translation           | Human evaluation / translation metrics |
| Distress detection    | Recall for critical cases              |
| Agent guidance        | Human QA rating                        |
| Latency               | P50 / P95                              |
| Audio quality         | Performance by SNR                     |
| Language              | Performance by language                |

Particular attention should be given to **false negatives for genuine distress** and false escalations that could unnecessarily interrupt normal service.

---

# 🧪 Testing Scenarios

A useful POC test set should contain examples such as:

```text
1. Calm customer
2. Frustrated customer
3. Angry but non-distressed customer
4. Financial hardship
5. Health emergency
6. Severe emotional distress
7. Threat of harm
8. Neutral conversation
9. Poor-quality audio
10. Background noise
11. Hindi speech
12. English speech
13. Mixed Hindi-English speech
14. Other multilingual speech
15. Very short clip
16. Unintelligible audio
```

The output should be reviewed by human evaluators rather than treating the model output as ground truth.

---

# 📁 Suggested Project Structure

```text
emotion-aware-calls/
│
├── emotion_aware_calls_poc.py
├── requirements.txt
├── README.md
│
└── .streamlit/
    └── secrets.toml
```

Example:

```text
.streamlit/secrets.toml
```

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

---

# 🔧 Configuration Points

The main configuration areas are:

### Gemini model catalog

```python
MODEL_CATALOG = {
    ...
}
```

### Default model

```python
DEFAULT_MODEL = "gemini-flash-lite-latest"
```

### Stress escalation threshold

```python
stress_score >= 7
```

### Audio minimum length

```python
0.3 seconds
```

### Recommended recording length

```text
3–10 seconds
```

These values can be changed during POC experimentation.

---

# 🧩 Error Handling

The application includes defensive handling for several failure scenarios.

### Missing optional packages

If `soundfile` or `librosa` is unavailable:

```text
Application continues
```

but local waveform/acoustic extraction is unavailable.

### Gemini unavailable

The application falls back to offline heuristic analysis.

### Invalid JSON

The application attempts to extract JSON from the model response.

If parsing still fails, a safe default response is returned rather than crashing the application.

### Short audio

Very short recordings do not receive meaningful local acoustic analysis.

---

# 📜 Model Response Parsing

Gemini is requested to return:

```text
application/json
```

The parser additionally protects against accidental markdown fences:

```text
```json
{ ... }
````

```

If valid JSON cannot be extracted, the application creates a safe fallback response.

---

# 🛡️ Design Principles

The POC follows several important principles:

### 1. Customer protection

The analysis should support de-escalation and appropriate assistance.

### 2. Human oversight

Critical situations should be routed to humans.

### 3. Explainability

The application exposes evidence and limitations instead of only showing a score.

### 4. Graceful degradation

The application can continue operating when the Gemini API is unavailable.

### 5. Explicit guardrails

Critical escalation rules are enforced in application code.

### 6. Multilingual support

The architecture does not assume that customers speak English.

### 7. Analytical honesty

The system explicitly distinguishes model-based analysis from crude offline acoustic heuristics.

---

# 📄 POC Disclaimer

> **This application is a proof of concept for demonstration and experimentation only. It is not connected to a production telephony or collections system. Voice-based emotion analysis should not be treated as a definitive assessment of a person's emotional or financial condition. Production deployment requires appropriate validation, consent/disclosure, privacy controls, security review, model governance, human oversight and regulatory/compliance approval.**

---

# 🔑 Gemini API Key

For development, obtain a Gemini API key through Google AI Studio.

The application expects:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
````

Do not commit the API key to source control.

For production, use the organization's approved secret-management and API-governance mechanisms.

---

# 📌 Quick Start

```bash
pip install -r requirements.txt
```

Configure:

```text
.streamlit/secrets.toml
```

with:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Then run:

```bash
streamlit run emotion_aware_calls_poc.py
```

Record a short call snippet:

```text
🎙️ Record → ⚡ Analyze this turn
```

Review:

```text
Stress
Emotion
Escalation
Agent guidance
Transcript
Translation
Acoustic characteristics
In-depth reasoning
Session timeline
```

---

# 🏦 BOB AI / BFSI Positioning

This POC demonstrates how **multimodal generative AI + local speech analytics + deterministic guardrails** can be combined into an agent-assist workflow for banking customer interactions.

The architecture separates:

```text
Signal Processing
       +
Generative AI
       +
Business Context
       +
Deterministic Guardrails
       +
Human Oversight
```

This separation provides a foundation for future experimentation with:

* AI-powered collections assistance
* Customer-service copilots
* Multilingual voice banking
* Contact-center intelligence
* Conversation QA
* Vulnerability-aware service workflows
* Real-time agent assistance
* AI-assisted escalation

---

# 📌 Summary

**Emotion-Aware Collections & Service Calls** is a Streamlit-based BFSI proof of concept that combines:

```text
🎙️ Browser Audio
       +
🔊 Local Acoustic Features
       +
🤖 Gemini Native Audio
       +
🌐 Multilingual Understanding
       +
📝 Transcription & Translation
       +
😟 Stress / Emotion Analysis
       +
🧠 Explainable Reasoning
       +
🚨 Deterministic Escalation Guardrails
       +
👨‍💼 Agent Guidance
       +
📈 Session Analytics
```

The central design goal is not simply to detect emotion, but to provide **responsible, explainable and human-supervised assistance during customer interactions**, particularly when a customer may require additional support.

---

## 📜 License

Add the appropriate Bank of Baroda / organization-specific license or internal POC usage statement before publishing the repository externally.
