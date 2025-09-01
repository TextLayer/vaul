from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vaul",
    version='1.1.1',
    description="A lightweight Python library for building agentic actions and workflows.",
    author="Spencer Porter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pydantic>=2.6.4',
        'pandas>=2.0.0',
        'tabulate>=0.9.0',
        'pyyaml>=6.0',
        'requests>=2.0',
        'mcp>=1.0.0',
        'nest-asyncio>=1.5.0',
    ],
    packages=find_packages(),
    python_requires='>=3.10',
)
