# Vaul

Vaul is a library designed to help developers create tool calls for AI systems, such as OpenAI's GPT models. It provides a simple and efficient way to define and manage these tool calls with built-in validation and schema generation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining Tool Calls](#defining-tool-calls)
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
from vaul import tool_call

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


# Define the tools that can be called and map them to the functions
tools = {
    'create_issue': create_issue,
    'get_issue': get_issue,
    'get_issues': get_issues,
    'update_issue': update_issue,
    'delete_issue': delete_issue,
}


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
    tools=[{
        "type": "function",
        "function": value.tool_call_schema
    } for key, value in tools.items()],
)

# Identify the tool call, if any
try:
    tool_call = response.choices[0].message.tool_calls[0].function.name
except AttributeError:
    tool_call = None
    
    
# Run the tool if it exists
if tool_call and tool_call in tools:
    tool_run = tools[tool_call].from_response(response, throw_error=False)
    print(tool_run)
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
