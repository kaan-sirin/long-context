# Kaan's Development Standards & Preferences

## Package Management
- **Python**: Use `uv` for package management
  - Add packages: `uv add <package_name>`
  - Install dependencies: `uv sync`
  - Run scripts: `uv run <script>`

## Environment Variables
- Always use `python-dotenv` for loading environment variables
- Store sensitive data in `.env` files
- Never commit `.env` files to version control

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

## LLM Integration
- **Primary**: OpenAI with GPT-5 models
- **Structured Outputs**: Use Pydantic models for type safety and validation
- **Best Practice**: Always use structured outputs for reliable data parsing

### OpenAI + Pydantic Example

```python
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# environment setup
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class TaskResponse(BaseModel):
    title: str
    priority: str
    estimated_hours: float
    dependencies: list[str]

# structured output request
response = client.beta.chat.completions.parse(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Break down this project into tasks"}],
    response_format=TaskResponse
)

result = response.choices[0].message.parsed
```

## Code Organization & Comments
- Use `#### section headers ############...` to separate major sections
- Use lowercase comments for inline explanations when logic isn't clear
- Keep comments concise and focus on "why" not "what"

```python
#### data processing section ###################################################

def process_user_input(raw_input):
    # normalize whitespace to handle copy-paste issues
    cleaned = raw_input.strip().lower()
    return cleaned

#### validation section ########################################################

def validate_email(email):
    # basic regex check - more robust validation happens server-side
    return '@' in email and '.' in email.split('@')[1]
```

## Prompt Engineering Standards
- **Format**: Always use markdown-structured prompts
- **Structure**: Role → Task → Tone/Guidelines → Examples/Context
- **Multi-line**: Use parentheses concatenation, never triple quotes for long prompts

### Prompt Template

```python
system_prompt = (
    "# Role\n"
    "You are a [specific role] with expertise in [domain].\n\n"
    "# Task\n" 
    "Your specific task description here.\n\n"
    "# Guidelines\n"
    "- Guideline 1\n"
    "- Guideline 2\n"
    "- Output format requirements\n\n"
    "# Example\n"
    "Example of expected output format...\n\n"
)

user_prompt = (
    f"# Context\n"
    f"**Key Info:** {variable}\n\n"
    f"**Requirements:** {requirements}\n\n"
    f"# Input Data\n"
    f"{input_data}\n\n"
)
```

## Git Workflow & Project Documentation

### Branching Strategy
- **main**: Production-ready code only
- **feature/[name]**: New features and major changes
- **fix/[name]**: Bug fixes and small improvements
- Always create branches for new features: `git checkout -b feature/youtube-processor`

### Commit Standards
- **Format**: Short, lowercase, descriptive messages
- **Examples**: "add youtube processor", "fix chunking overlap", "update timestamp parsing"
- **Frequency**: Commit early and often - after each logical unit of work
- **Push regularly**: Keep remote updated for backup and collaboration

### Project Documentation
- **PROJECT_STATUS.md**: Maintain current progress and next steps
- **Update continuously**: After each significant change or feature completion
- **Format**: Simple markdown - what's done, what's in progress, what's next
- **Purpose**: Enable easy handoff between development sessions/developers

### Merge Process
- Test feature thoroughly before merging
- Ask user for merge approval on significant features
- Delete feature branches after successful merge
- Update PROJECT_STATUS.md after merges

## General Principles
- Try to keep each file under 300 lines
- Use absolute imports when possible
- Keep functions small and focused
- Test early and often with simple integration tests