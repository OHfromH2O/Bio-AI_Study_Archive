# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is **Study 03** in the "Vibe coding 실습" learning series:
- Study 01: MNIST digit classification desktop app (Python, TensorFlow, Tkinter)
- Study 02: Web-based todo application (HTML/CSS/JavaScript)
- Study 03: This project — LLM-integrated application using OpenRouter API (in development)

The `.env` file contains an `OPENROUTER_API_KEY` for use with the OpenRouter API, which provides access to multiple LLM providers (OpenAI, Anthropic, etc.) through a unified endpoint.

## Environment

API credentials are stored in `.env` and should be loaded via a library such as `python-dotenv` (Python) or `dotenv` (Node.js). Never hardcode the key in source files.
