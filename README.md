# Grant Proposal Review Environment

> **OpenEnv Hackathon — Meta × Scaler | April 2026**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.ai)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

The **Grant Proposal Review Environment** is an OpenEnv-compliant reinforcement learning environment where an AI agent learns to evaluate research grant proposals the way a real funding committee does.

The agent interacts with a simulated applicant, requests proposal sections strategically, detects hidden weaknesses, and makes funding decisions — all under partial observability and a limited action budget.

> **We are not training an agent. We are building the world it learns in.**
> Think of it like building the rules of chess, not the player.

---

## Motivation

Grant evaluation is a genuinely hard real-world task. It requires:

- Reasoning under incomplete information
- Strategic questioning to surface hidden weaknesses
- Decision-making under uncertainty with limited resources

Unlike single-step evaluation tasks, our environment is **multi-turn** and **partially observable**. Weaknesses in proposals remain hidden unless the agent asks exactly the right questions — making this a meaningful benchmark for agent reasoning, not just classification.

Real-world utility is the primary judging criterion (30% weight). Grant review is something AI labs and funding institutions genuinely want to automate and evaluate agents on.

---

## Key Design Features

**Partial Observability**
The agent starts with only the abstract. Methodology, budget, team composition, and references are locked behind explicit action requests. The agent never sees the full picture unless it asks.

**Strategic Applicant Simulation**
The applicant is not cooperative. It answers what is asked but never volunteers weaknesses. Hidden flaws only surface if the agent probes the right areas with the right questions.

**Dense Reward Shaping**
Rewards are not binary end-of-episode signals. The agent receives partial credit throughout for information gain, flaw detection, and efficient decision-making — providing meaningful training signal at every step.

**Deterministic Graders**
All three tasks use programmatic graders with clear, reproducible scoring criteria. Same input always produces the same score.

---

## Environment Design

### Episode Flow

```
reset()
  └── Agent receives: title, abstract, requested amount, evaluation rubric

step(action) × N
  ├── request_methodology    → unlocks methodology section
  ├── request_budget         → unlocks budget breakdown
  ├── request_team           → unlocks team composition
  ├── request_references     → unlocks references
  ├── request_clarification  → applicant responds strategically
  ├── approve                → episode ends, grader scores decision
  └── reject                 → episode ends, grader scores decision

state()
  └── Returns full environment state including ground truth (for graders)
```

### Observation Space

| Field | Visibility | Description |
|-------|-----------|-------------|
| `abstract` | Always | Proposal abstract |
| `title` | Always | Proposal title |
| `requested_amount` | Always | Funding requested (USD) |
| `methodology` | On request | Research methodology — may contain hidden flaws |
| `budget_breakdown` | On request | Line-item budget — may contain conflict of interest |
| `team_composition` | On request | Team members and roles — cross-reference with budget |
| `references` | On request | Supporting literature |
| `clarification_response` | After request | Applicant's strategic answer |
| `actions_remaining` | Always | Remaining action budget |
| `evaluation_criteria` | Always | Rubric weights |

### Action Space

| Action | Description |
|--------|-------------|
| `request_methodology` | Unlock methodology section |
| `request_budget` | Unlock budget breakdown |
| `request_team` | Unlock team composition |
| `request_references` | Unlock references |
| `request_clarification` | Ask applicant a specific question |
| `approve` | Fund the proposal — ends episode |
| `reject` | Reject the proposal — ends episode |

### Reward Structure

| Signal | Value | Trigger |
|--------|-------|---------|
| Relevant information gain | +0.15 | Unlocked section containing hidden signal |
| Flaw detected via cross-reference | +0.25 | Both budget and team unlocked (hard task) |
| Correct funding decision | +1.00 | Approve/reject matches ground truth |
| Confidence bonus | +0.20 | High confidence + correct decision |
| Redundant action | -0.05 | Requesting already-unlocked section |
| Forced termination | -0.10 | Action budget exhausted without decision |
| Wrong decision | -0.50 | Approve/reject contradicts ground truth |

---

## Tasks

### Easy — *Complete Proposal*
A well-formed, fundable proposal for an AI-assisted diabetic retinopathy detection system in rural clinics. No hidden flaws. Agent should apply the rubric and approve.

**Correct decision:** APPROVE
**Key challenge:** Efficient evaluation — don't waste actions on unnecessary sections.

---

### Medium — *Hidden Methodology Flaw*
A privacy-preserving mental health monitoring system using federated learning. Strong abstract, but the methodology has a critical labeling gap — no ground truth between monthly clinical assessments.

**Correct decision:** REJECT
**Key challenge:** Agent must request and carefully read the methodology section to find the flaw.

---

### Hard — *Conflict of Interest*
A neuromorphic drone navigation system. Technically impressive. But the PI holds 40% equity in a private company that receives a $24,000 technology transfer fee from the grant budget.

**Correct decision:** REJECT
**Key challenge:** The conflict is only detectable by requesting **both** the budget (sees the fee) AND the team (sees the equity stake) and cross-referencing. Neither section alone reveals it.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Environment framework | OpenEnv |
| Typed models | Pydantic v2 |
| API layer | FastAPI + uvicorn |
| Baseline agent | OpenAI client → HuggingFace LLM |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |

---

## Project Structure

```
grant-review-env/
├── inference.py              # Baseline inference script (root — mandatory)
├── environment/
│   ├── __init__.py
│   ├── env.py                # Main GrantReviewEnv class
│   ├── models.py             # Pydantic observation/action/reward models
│   ├── applicant.py          # Strategic applicant simulation
│   └── graders/
│       ├── __init__.py
│       ├── easy.py           # Deterministic grader — easy task
│       ├── medium.py         # Deterministic grader — medium task
│       └── hard.py           # Deterministic grader — hard task
├── tasks/
│   ├── task_easy.json        # Complete well-formed proposal
│   ├── task_medium.json      # Proposal with methodology flaw
│   └── task_hard.json        # Proposal with hidden conflict of interest
├── openenv.yaml              # OpenEnv spec configuration
├── Dockerfile                # Container definition
├── requirements.txt
└── README.md
```

---

## Setup and Usage

### Prerequisites

- Python 3.11+
- Docker (for deployment)
- HuggingFace account and API token

### Installation

```bash
git clone https://github.com/aiyu221b3/grant-review-env
cd grant-review-env
pip install -r requirements.txt
```

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
```

### Run Baseline Inference

```bash
python inference.py
```

Expected output format:
```
[START] task=easy env=grant-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=request_methodology reward=0.15 done=false error=null
[STEP] step=2 action=approve reward=1.20 done=true error=null
[END] success=true steps=2 score=0.732 rewards=0.15,1.20
```

### Test Environment Without LLM

```python
from environment import GrantReviewEnv, GrantReviewAction, ActionType

env = GrantReviewEnv(task_name="easy")
result = env.reset()
print("Abstract:", result.observation.abstract[:100])

action = GrantReviewAction(action_type=ActionType.REQUEST_METHODOLOGY)
result = env.step(action)
print("Reward:", result.reward)

action = GrantReviewAction(
    action_type=ActionType.APPROVE,
    justification="Strong proposal.",
    confidence=0.9
)
result = env.step(action)
print("Final reward:", result.reward, "| Done:", result.done)

from environment.graders import grade_easy
print("Grader score:", grade_easy(env.state()))
```

### Docker

```bash
docker build -t grant-review-env .
docker run -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  grant-review-env
```

### OpenEnv Validation

```bash
pip install openenv
openenv validate .
```

---

## Baseline Scores

| Task | Steps | Score | Success |
|------|-------|-------|---------|
| Easy | 5 | 0.646 | ✅ |
| Medium | 5 | 0.567 | ✅ |
| Hard | 4 | 0.000 | ❌ |

*Baseline agent: Qwen/Qwen2.5-7B-Instruct via HuggingFace router.*

The hard task failure is expected and intentional — the baseline agent detects the evidence (requests both budget and team) but fails to connect the conflict of interest to a reject decision. This demonstrates the environment's difficulty progression and creates room for stronger agents to differentiate.

---

## Team

| Name | Role |
|------|------|
| **Ayushi Bhattacharya** | Environment architecture, Pydantic models, observation space design, partial observability mechanics, reward shaping, strategic applicant simulation |
| **Sohan** | Grant evaluation rubric, deterministic graders, Docker containerization, HuggingFace Spaces deployment |
| **Sourashree** | Grant proposal design (all three tasks), README documentation, demo narrative |

*Built for the OpenEnv Hackathon — Meta × Scaler, April 2026.*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
