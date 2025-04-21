# Vaul

Vaul is a library designed to help developers create tool calls for AI systems, such as OpenAI's GPT models. It provides a simple and efficient way to define and manage these tool calls with built-in validation and schema generation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining Tool Calls](#defining-tool-calls)
  - [Managing Tool Calls with Toolkit](#managing-tool-calls-with-toolkit)
  - [Tool Documentation Format](#tool-documentation-format)
    - [Keeping System Prompts in Sync with Toolkits](#keeping-system-prompts-in-sync-with-toolkits)
  - [Interacting with OpenAI](#interacting-with-openai)
  - [More Complex Examples](#more-complex-examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Vaul is designed to simplify the process of creating tool calls that can be used by AI systems, such as OpenAI's GPT models. With Vaul, developers can easily define functions with validation, generate schemas, and integrate these tool calls with AI systems.

## Features

- **Easy Tool Call Definition**: Define tool calls using simple decorators.
- **Automatic Schema Generation**: Generate OpenAPI schemas from function signatures.
- **Built-in Validation**: Ensure input data integrity using Pydantic validation.
- **Seamless AI Integration**: Integrate tool calls with AI systems like OpenAI's GPT.
- **Toolkit Management**: Organize and manage collections of tools with the Toolkit class.
- **Documentation Generation**: Create beautiful markdown tables of your tools with `to_markdown()` to keep system prompts in sync with your toolkit.
- **Customizable**: Define custom actions and manage them with ease.

## Installation

To install Vaul, you can use `pip`:

```bash
pip install vaul
```

## Usage

### Defining Tool Calls

Vaul allows you to define tool calls using simple decorators. Here is an example of how to define a function that can be utilized by an AI system:

```python
from vaul import tool_call

@tool_call
def add_numbers(a: int, b: int) -> int:
    return a + b
```

### Managing Tool Calls with Toolkit

Vaul provides a `Toolkit` class that helps you organize and manage multiple tool calls efficiently:

```python
from vaul import Toolkit, tool_call

# Create a toolkit to manage your tools
toolkit = Toolkit()

@tool_call
def add_numbers(a: int, b: int) -> int:
    """Add two numbers
    
    Desc: Adds two numbers together.
    Usage: When you need to calculate the sum of two numbers.
    """
    return a + b

@tool_call
def multiply_numbers(a: int, b: int) -> int:
    """Multiply numbers
    
    Desc: Multiplies two numbers together.
    Usage: When you need to calculate the product of two numbers.
    """
    return a * b

@tool_call
def subtract_numbers(a: int, b: int) -> int:
    """Subtract numbers
    
    Desc: Subtracts the second number from the first.
    Usage: When you need to calculate the difference between two numbers.
    """
    return a - b
    
# Register a single tool
toolkit.add(add_numbers)

# Or register multiple tools at once
toolkit.add_tools(multiply_numbers, subtract_numbers)

# Generate schemas for all tools
tool_schemas = toolkit.tool_schemas()

# Execute a specific tool by name
result = toolkit.run_tool("add_numbers", {"a": 5, "b": 3})
print(result)  # Output: 8

# Access all tool names
print(toolkit.tool_names)  # Output: ['add_numbers', 'multiply_numbers', 'subtract_numbers']

# Generate a markdown table of all tools
markdown_table = toolkit.to_markdown()
print(markdown_table)
# Output:
# ### Tools
# | Tool | Description | When to Use |
# |------|-------------|-------------|
# | `add_numbers` | Adds two numbers together. | When you need to calculate the sum of two numbers. |
# | `multiply_numbers` | Multiplies two numbers together. | When you need to calculate the product of two numbers. |
# | `subtract_numbers` | Subtracts the second number from the first. | When you need to calculate the difference between two numbers. |
```

### Tool Documentation Format

When creating tool calls, you can add structured documentation to your function docstrings that will be extracted by the `to_markdown` method. This makes it easy to generate clear documentation tables for users.

The docstring format supports the following special tags:

- `Desc:` - A detailed description of what the tool does
- `Usage:` - Guidance on when to use this tool

Example of a well-documented tool:

```python
@tool_call
def search_database(query: str, limit: int = 10) -> List[Dict]:
    """Search Database
    
    Desc: Performs a semantic search against the knowledge database.
    Usage: Use this when you need to find information about a specific topic or question.
    """
    # Implementation here
    ...
```

If no `Desc:` tag is provided, the first line of the docstring will be used as the description.

You can then generate a nicely formatted markdown table of all your tools using:

```python
markdown_table = toolkit.to_markdown()
```

This will produce a table like:

```markdown
### Tools
| Tool | Description | When to Use |
|------|-------------|-------------|
| `search_database` | Performs a semantic search against the knowledge database. | Use this when you need to find information about a specific topic or question. |
```

#### Keeping System Prompts in Sync with Toolkits

One of the most powerful features of `to_markdown` is its ability to help maintain consistency between your code and AI system prompts. As your toolkit evolves with new tools or updated functionality, you can dynamically generate up-to-date documentation to include in your system prompts.

For example, when working with LLM agents that need to know about available tools:

```python
# Register all your tools to the toolkit
toolkit.add_tools(search_database, create_document, update_settings)

# Generate the tools documentation table
tools_documentation = toolkit.to_markdown()

# Use this in your system prompt
system_prompt = f"""You are a helpful assistant with access to the following tools:

{tools_documentation}

When a user asks a question, use the most appropriate tool based on the 'When to Use' guidance.
Always prefer using tools over making up information.
"""

# Create your chat completion
response = openai_session.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ],
    tools=toolkit.tool_schemas()
)
```

This approach ensures that:
1. Your system prompt always contains the latest tools information
2. The AI has accurate guidance on when to use each tool
3. The tool descriptions in the prompt match the actual implementation
4. When you add, remove or modify tools, the system prompt updates automatically

This synchronization is essential for maintaining consistency in agent behavior and preventing the confusion that happens when system prompts describe tools differently than they're actually implemented.

### Interacting with OpenAI

You can integrate Vaul with OpenAI to create, monitor, and deploy tool calls. Here is an example that demonstrates how to use a tool call with OpenAI's GPT-3.5-turbo:

```python
import os

from openai import OpenAI
from vaul import tool_call

openai_session = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

@tool_call
def add_numbers(a: int, b: int) -> int:
    return a + b

response = openai_session.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is 2 + 2?"
        }
    ],
    temperature=0.7,
    top_p=1,
    stream=False,
    seed=42,
    tools=[{
        "type": "function",
        "function": add_numbers.tool_call_schema
    }],
    tool_choice={
        "type": "function",
        "function": {
            "name": add_numbers.tool_call_schema["name"],
        }
    }
)

print(response.choices[0].message.model_dump(exclude_unset=True))

# Output:
# {'content': None, 'role': 'assistant', 'tool_calls': [{'id': 'call_xxxxxxxxxxxxxx', 'function': {'arguments': '{"a":2,"b":2}', 'name': 'add_numbers'}, 'type': 'function'}]}

# Run the function call
print(add_numbers.from_response(response))

# Output:
# 4
```

### More Complex Examples
Let's take a look at how you might handle a more complex application, such as one that integrates multiple potential tool calls:

```python
import os

from jira import JIRA
from openai import OpenAI
from vaul import tool_call, Toolkit

from dotenv import load_dotenv

load_dotenv('.env')

jira = JIRA(
    server=os.environ.get("JIRA_URL"),
    basic_auth=(
        os.environ.get("JIRA_USER"),
        os.environ.get("JIRA_API_TOKEN")
    )
)

openai_session = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create a toolkit to manage all our Jira-related tools
toolkit = Toolkit()

@tool_call
def create_issue(summary: str, description: str, issue_type: str) -> dict:
    """
    Creates a Jira issue.
    :param summary: The issue summary
    :param description: The issue description
    :param issue_type: The issue type
    :return: The created issue
    """
    try:
        new_issue = jira.create_issue(
            fields={
                "project": {"key": "KAN"},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type}
            }
        )
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "id": new_issue.id,
        "key": new_issue.key,
        "summary": new_issue.fields.summary,
        "description": new_issue.fields.description,
        "type": new_issue.fields.issuetype.name
    }


@tool_call
def get_issue(issue_id: str) -> dict:
    """
    Gets a Jira issue by ID.
    :param issue_id: The issue ID
    :return: The issue
    """
    try:
        issue = jira.issue(issue_id)
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "id": issue.id,
        "key": issue.key,
        "summary": issue.fields.summary,
        "description": issue.fields.description,
        "type": issue.fields.issuetype.name
    }


@tool_call
def get_issues(project: str) -> dict:
    """
    Gets all issues in a project.
    :param project: The project key
    :return: The issues
    """
    try:
        issues = jira.search_issues(f"project={project}")
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "issues": [
            {
                "id": issue.id,
                "key": issue.key,
                "summary": issue.fields.summary,
                "description": issue.fields.description,
                "type": issue.fields.issuetype.name
            } for issue in issues
        ]
    }


@tool_call
def update_issue(issue_id: str, summary: str, description: str, issue_type: str) -> dict:
    """
    Updates a Jira issue.
    :param issue_id: The issue ID
    :param summary: The issue summary
    :param description: The issue description
    :param issue_type: The issue type
    :return: The updated issue
    """
    try:
        issue = jira.issue(issue_id)

        fields = {
            "summary": summary if summary else issue.fields.summary,
            "description": description if description else issue.fields.description,
            "issuetype": {"name": issue_type if issue_type else issue.fields.issuetype.name}
        }

        issue.update(fields=fields)
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "id": issue.id,
        "key": issue.key,
        "summary": issue.fields.summary,
        "description": issue.fields.description,
        "type": issue.fields.issuetype.name
    }


@tool_call
def delete_issue(issue_id: str) -> dict:
    """
    Deletes a Jira issue.
    :param issue_id: The issue ID
    """
    try:
        jira.issue(issue_id).delete()
    except Exception as e:
        return {
            "error": str(e)
        }

    return {
        "message": "Issue deleted successfully"
    }


# Register all tools with the toolkit using the bulk add method
toolkit.add_tools(
    create_issue, 
    get_issue, 
    get_issues, 
    update_issue, 
    delete_issue
)

# Send a message to the OpenAI API to create a new issue
response = openai_session.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a Jira bot that can create, update, and delete issues. You can also get issue details, transitions, and comments."
        },
        {
            "role": "user",
            "content": "Create a new issue with the summary 'Test Issue' and the description 'This is a test issue' of type 'Task'."
        }
    ],
    tools=toolkit.tool_schemas(),  # Get schemas for all tools in the toolkit
)

# Identify the tool call, if any
try:
    tool_name = response.choices[0].message.tool_calls[0].function.name
    tool_arguments = response.choices[0].message.tool_calls[0].function.arguments
except (AttributeError, IndexError):
    tool_name = None
    
# Run the tool if it exists
if tool_name:
    # Get the tool from toolkit and run it
    try:
        import json
        arguments = json.loads(tool_arguments)
        result = toolkit.run_tool(tool_name, arguments)
        print(result)
    except ValueError as e:
        print(f"Error running tool: {e}")
```


## Roadmap
- [ ] Add support for other providers (e.g. Anthropic, Cohere, etc.)
- [ ] Add examples for parallel tool calls
- [ ] Better error handling and logging
- [ ] Improved support for types and defaults

## Contributing

We welcome contributions from the community! If you would like to contribute to Vaul, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them to your branch.
4. Push your changes to your fork.
5. Create a pull request to the main repository.
6. We will review your changes and merge them if they meet our guidelines.
7. Thank you for contributing to Vaul!

## License

Vaul is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

### Note on Inspiration

Vaul was created as a way to build on the simplicity and developer experience provided by Jason Liu's excellent `Instructor` package (formerly `openai-function-call`), after the decorator functionality was removed in newer versions. The goal is to maintain the ease of defining and using tool calls for AI systems, ensuring a smooth developer experience.

If you haven't seen `Instructor` before, I highly recommend checking it out if you're working with structured outputs for AI systems:

- [Instructor on GitHub](https://github.com/jxnl/instructor)
- [Instructor Documentation](https://python.useinstructor.com/)
