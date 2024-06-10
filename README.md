# Vaul

Vaul is a library designed to help developers create tool calls for AI systems, such as OpenAI's GPT models. It provides a simple and efficient way to define and manage these tool calls with built-in validation and schema generation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining Tool Calls](#defining-tool-calls)
  - [Interacting with OpenAI](#interacting-with-openai)
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

Vaul is licensed under the GNU General Public License v3.0. See the LICENSE file for more information.

### Note on Inspiration

Vaul was created as a way to bring back some of the simplicity and developer experience provided by Jason Liu's `Instructor` package (formerly `openai-function-call`), after specific functionality was removed in newer versions. The goal is to maintain the ease of defining and using tool calls for AI systems, ensuring a smooth developer experience.

If you haven't seen `Instructor` before, I highly recommend checking it out:

- [Instructor on GitHub](https://github.com/jxnl/instructor)
- [Instructor Documentation](https://python.useinstructor.com/)
