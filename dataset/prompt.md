# PDF to JSON Conversion Guide

This document outlines the steps to parse a PDF document and convert its content into a structured JSON format. The PDF contains multiple-choice questions, short-answer questions, and solutions. The goal is to extract questions, their types, options (if applicable), and format them into a JSON object. Additionally, mathematical formulas, equations, and expressions should be identified and wrapped in MathML syntax enclosed within `$$` delimiters.

## Steps

### 1. Identify Sections
The PDF is divided into sections (e.g., "Section A. Multiple choices" or "Section B. Short-answer Questions"). Each section contains questions and solutions.

### 2. Extract Questions
For each question, extract the following details:

- **Question Text**: The full text of the question, including sub-questions (e.g., part a, part b).
- **Question Type**: Determine if the question is "multiple-choice" or "short-answer."
- **Options**: For multiple-choice questions, extract all options (e.g., A, B, C, D, E). For other questions, set options to `null`.

### 3. Mathematical Formulas and Equations
Identify all mathematical formulas, equations, and expressions. For each mathematical formula or equation, wrap it in MathML syntax and enclose it within `$$` delimiters. Ensure that the MathML representation is accurate and properly formatted.

### Sample JSON Structure

```json
{
  "question": "The full text of the question",
  "type": "multiple-choice" or "short-answer",
  "options": ["Option A", "Option B", "Option C", ...] (null if not applicable)
}