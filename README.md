# Generative AI with Python Examples

A collection of Jupyter notebooks demonstrating the use of **LlamaIndex**, **LangChain**, **Ollama**, and the **Transformers** library for building generative AI applications in Python.

## Table of Contents

- [Introduction](#introduction)
- [Notebooks](#notebooks)
    - [Transformers library examples](#transformers-library-examples)
    - [LangChain examples](#langchain-examples)
    - [LlamaIndex examples](#llamaindex-examples)
    - [Ollama examples](#ollama-examples)
    - [Vector database](vector_database.ipynb)
    - [Evaluation ROUGE/BLEU](evaluation_rouge_bleu.ipynb)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Slides](#slides)
- [Project: AI chatbot with RAG](#Project)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository contains Jupyter notebooks that provide practical examples of using various frameworks for generative AI in Python. The notebooks are designed to help you understand and apply these tools in your own projects.

## Notebooks

### Transformers Library Examples

1. **[transformers_llms_usa_case_types.ipynb](transformers_llms_usa_case_types.ipynb)**

    - **Description**: Common examples of using transformers library with models from huggingface.co.

2. **[transformers_parameters.ipynb](transformers_parameters.ipynb)**

    - **Description**: This notebook shows usage of transformers library for same models with different values of parameters like temperature, top_k, no_repeat_ngram_size and others.

3. **[transformers_few_shot.ipynb](transformers_few_shot.ipynb)**

    - **Description**: This notebook shows how to add examples to the prompt.
      LLM provided with examples can improve significantly results.

4. **[transformers_fine_tune.ipynb](transformers_fine_tune.ipynb)**

    - **Description**: This notebook shows how to fine tune LLM model. Fine-tuning is advanced technique which can modify model parameters to perform better in specific task.

### LangChain Examples

1. **[langchain_basic_examples.ipynb](langchain_basic_examples.ipynb)**

    - **Description**: "Get started" examples of using LLMs with LangChain framework.

2. **[langchain_chain.ipynb](langchain_chain.ipynb)**

    - **Description**: Chaining LLM tasks helps building advanced AI applications that can handle a sequence of tasks or resoning.

3. **[langchain_templates_and_output_parsers.ipynb](langchain_templates_and_output_parsers.ipynb)**

    - **Description**: This notebook presents using LangChain prompt templates and output parsers.
      Prompt templates help to structurize prompt and better instruct LLM to deal with specific tasks.
      Output parsers are used to ensure type of return values and formatting is always as expected.

4. **[langchain_memory.ipynb](langchain_memory.ipynb)**

    - **Description**: Storing and summarizing conversation history in a structurized form.

5. **[langchain_evaluation.ipynb](langchain_evaluation.ipynb)**

    - **Description**: This notebook presents LangChain criteria evaluators which help to evaluate generated output with other LLM using defined categories.

6. **[langchain_agents.ipynb](langchain_agents.ipynb)**

    - **Description**: Agents use LLM to determine what actions to perform and in what order. Agents can use set of predefined tools to solve complex tasks.


### LlamaIndex Examples

- **Notebook**: [llamaindex_example.ipynb](llamaindex_example.ipynb)
- **Description**: This notebook shows how to run Llamaindex framework locally to create virtual AI assistant based on RAG (Retrieval Augmented Generation).

### Ollama Examples

- **Notebook**: [ollama_example.ipynb](ollama_example.ipynb)
- **Description**: Ollama is a tool for using LLMs on local environment via API.
  This gives simplicity and flexibility for creating AI/LLM/RAG based applications.

### Vector database

- **Notebook**: [vector_database.ipynb](vector_database.ipynb)
- **Description**: This notebook presents how to use embeddings and store them in vector database.

### Evaluation ROUGE / BLEU / METEOR 

- **Notebook**: [evaluation_rouge_bleu.ipynb](evaluation_rouge_bleu.ipynb)
- **Description**: This notebook presents evaluation techniques for LLMs output such as ROUGE, BLEU and METEOR.
All presented metrics are based on comparison of words in AI generated text with human provided result text.

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Installed libraries as specified in `requirements.txt`

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/rzarno/course-generative-ai-python.git
cd course-generative-ai-python
pip install -r requirements.txt
```

## Usage

Open the notebooks using Jupyter:

```bash
jupyter lab
```

Navigate to the notebook you wish to explore and run the cells sequentially.

## Slides

Find definitions and examples of working with LLMs in Python on presentations in [SLIDES](slides) dir. Currently available in Polish (translation to en in progress).

[1. Architecture and types of large language models.pdf](slides/1.%20Architecture%20and%20types%20of%20large%20language%20models.pdf)\
[2. Usage of LLM and RAG - case study.pdf](slides/2.%20Usage%20of%20LLM%20and%20RAG%20-%20case%20study.pdf)\
[3. Practising with LLMs, prompt engineering.pdf](slides/3.%20Practising%20with%20LLMs%2C%20prompt%20engineering.pdf)\
[4. Frameworks for LLMs - LangChain, Ollama, LlamaIndex, HuggingFace.pdf](slides/4.%20Frameworks%20for%20LLMs%20-%20LangChain%2C%20Ollama%2C%20LlamaIndex%2C%20HuggingFace.pdf)\
[5. RAG - Retrieval Augmented Generation.pdf](slides/5.%20RAG%20-%20Retrieval%20Augmented%20Generation.pdf)\
[6. LLM Results evaluation.pdf](slides/6.%20LLM%20Results%20evaluation.pdf)\
[7. Parameter-Efficient Fine-Tuning (PEFT), LoRA i RLHF.pdf](slides/7.%20Parameter-Efficient%20Fine-Tuning%20%28PEFT%29%2C%20LoRA%20i%20RLHF.pdf)

## Project

AI chatbot project in Streamlit framework [project-ai-chatbot-rag-langchain](project-ai-chatbot-rag-langchain)

based on https://github.com/shashankdeshpande/langchain-chatbot

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE.txt) file for details.

## Contact

For any questions or feedback, please open an issue or contact [rzarno](https://github.com/rzarno).