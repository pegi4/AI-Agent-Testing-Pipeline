# AI Agent Testing Pipeline

A complete demonstration of automated testing for AI agents using Langfuse for observability and evaluation.

## 🎯 Project Purpose

This project demonstrates how to:
1. **Build an AI agent** with RAG (Retrieval-Augmented Generation) and tool calling
2. **Observe agent behavior** using Langfuse tracing
3. **Automatically test agent correctness** using Langfuse's Experiment Runner and DeepEval
4. **Monitor and evaluate** agent performance in a unified dashboard

### The Burger Shop Agent

A simple but complete AI agent that:
- Uses **RAG** (FAISS vector store) to look up menu prices
- Uses **Tools** to place orders
- Integrates with **Langfuse** for full observability
- Can be tested with **automated evaluation** frameworks

## 📋 Prerequisites

- Python 3.11+
- Google Gemini API key
- Docker and Docker Compose (only if self-hosting Langfuse)

## 🚀 Setup Instructions

### 1. Clone and Navigate

```bash
cd main/
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the `main/` directory:

```bash
# Google Gemini API Key (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Langfuse Configuration (required)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=http://localhost:3000
```

**Get your Gemini API key**: https://aistudio.google.com/apikey

### 5. Set Up Langfuse

You have two options for Langfuse:

#### Option A: Langfuse Cloud (Recommended for Quick Start)

Langfuse Cloud is a fully managed solution hosted by the Langfuse team.

1. Sign up at https://cloud.langfuse.com
2. Create a new project
3. Go to Settings → API Keys
4. Copy your `Public Key` and `Secret Key` to your `.env` file
5. Set `LANGFUSE_HOST=https://cloud.langfuse.com` in your `.env`

**Benefits:**
- No setup required
- Managed infrastructure
- Automatic updates
- Free tier available

#### Option B: Self-Hosted Langfuse (Local Development)

For local development or self-hosting, use Docker Compose:

```bash
cd ../langfuse
docker-compose up -d
```

Wait for all services to start (about 30-60 seconds), then:

1. Open http://localhost:3000 in your browser
2. Create an account (first-time setup)
3. Create a new project
4. Go to Settings → API Keys
5. Copy your `Public Key` and `Secret Key` to your `.env` file
6. Ensure `LANGFUSE_HOST=http://localhost:3000` in your `.env`

**Note:** Self-hosting requires Docker and sufficient system resources. For production deployments, see [Langfuse Self-Hosting Documentation](https://langfuse.com/self-hosting).

### 6. Verify Setup

Run a quick test:

```bash
python burger_agent.py
```

You should see:
- Agent response printed to console
- Trace ID (if Langfuse is connected)
- Check Langfuse dashboard:
  - **Cloud**: https://cloud.langfuse.com → Tracing
  - **Self-hosted**: http://localhost:3000 → Tracing

## 🧪 Testing

### Option 1: Langfuse Native Testing (Recommended)

Uses Langfuse's built-in Experiment Runner:

```bash
python test_langfuse_native.py
```

Or with pytest:

```bash
pytest test_langfuse_native.py -v
```

**What it does:**
- Runs 4 test cases through the agent
- Evaluates outputs with custom evaluators
- Sends results to Langfuse dashboard
- View results: Langfuse → Datasets → Experiments

### Option 2: DeepEval Testing

Uses DeepEval framework for LLM-as-a-Judge evaluation:

```bash
python test_agent.py
```

Or with pytest:

```bash
pytest test_agent.py -v
```

**What it does:**
- Runs 4 test cases
- Uses Gemini LLM to evaluate correctness
- Tests: RAG accuracy, tool execution, tone

### Option 3: DeepEval + Langfuse Integration

Sends DeepEval scores to Langfuse:

```bash
python test_agent_with_langfuse.py
```

## 📊 Viewing Results

### In Langfuse Dashboard

Access your dashboard:
- **Cloud**: https://cloud.langfuse.com
- **Self-hosted**: http://localhost:3000

1. **Tracing** → `/traces`
   - See every agent execution
   - View tool calls, LLM requests, token usage, costs
   - Debug agent behavior

2. **Datasets** → `/datasets`
   - View experiment runs
   - See test case results
   - Compare runs over time

3. **Scores** → `/scores`
   - View evaluation scores
   - Track quality metrics over time

## 📁 Project Structure

```
main/
├── burger_agent.py              # The AI agent (RAG + Tools)
├── test_langfuse_native.py      # Langfuse native testing
├── test_agent.py                # DeepEval testing
├── test_agent_with_langfuse.py  # DeepEval + Langfuse integration
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
└── README.md                    # This file
```

## 🔧 How It Works

### The Agent (`burger_agent.py`)

1. **RAG System**: FAISS vector store with menu data
   - "Big Mac: $5"
   - "Whopper: $6"
   - "Fries: $2"

2. **Tools**:
   - `lookup_price(query)`: Searches vector store for prices
   - `place_order(items)`: Returns order confirmation

3. **LLM**: Google Gemini 2.0 Flash
4. **Observability**: Langfuse CallbackHandler captures all interactions

### Testing Approaches

**Langfuse Native** (`test_langfuse_native.py`):
- Uses `langfuse.run_experiment()`
- Custom evaluators (price accuracy, tool execution)
- Results stored in Langfuse automatically

**DeepEval** (`test_agent.py`):
- Uses `GEval` (LLM-as-a-Judge)
- Gemini evaluates if outputs are correct
- Semantic evaluation, not exact matching

## 🎓 Key Concepts

### Observability vs Testing

- **Observability (Langfuse Tracing)**: See what happened
  - Tool calls, LLM requests, costs, latency
  - Always running, captures everything

- **Testing (Evaluation)**: Verify if it's correct
  - Automated checks against test cases
  - Pass/fail results
  - Run on-demand or in CI/CD

### Why Both?

- **Tracing** helps you debug when things go wrong
- **Testing** catches regressions before production
- Together: Complete visibility and quality assurance

## 🐛 Troubleshooting

### Langfuse not connecting?

**If using Langfuse Cloud:**
1. Verify `LANGFUSE_HOST=https://cloud.langfuse.com` in `.env`
2. Check your API keys are correct
3. Verify your account is active at https://cloud.langfuse.com

**If using self-hosted:**
1. Check Docker is running: `docker ps`
2. Verify Langfuse is up: http://localhost:3000
3. Check `.env` has correct keys and `LANGFUSE_HOST=http://localhost:3000`
4. Restart Langfuse: `cd ../langfuse && docker-compose restart`
5. Check logs: `cd ../langfuse && docker-compose logs`

### Tests failing?

1. Verify `GEMINI_API_KEY` is set in `.env`
2. Check you have API credits/quota
3. Run `python burger_agent.py` first to verify agent works

### Import errors?

1. Make sure virtual environment is activated
2. Reinstall: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.11+)

## 📚 Additional Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Self-Hosting Guide](https://langfuse.com/self-hosting)
- [Langfuse Cloud](https://cloud.langfuse.com)
- [LangChain Documentation](https://docs.langchain.com)
- [DeepEval Documentation](https://docs.deepeval.io)
- [Google Gemini API](https://ai.google.dev)

## 👤 For Niko

This project demonstrates a complete AI agent testing pipeline suitable for production use. The setup includes:

1. **Production-ready observability** with Langfuse
2. **Automated testing** with multiple frameworks
3. **CI/CD integration** ready (pytest compatible)
4. **Scalable architecture** that can handle hundreds of test cases

The code is clean, well-documented, and follows best practices for LLM application development.

## 📝 License

This is a demonstration project for educational purposes.

