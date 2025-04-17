#!/usr/bin/env python
import os
import sys
import json
import openai

def convert_to_json_with_openai(
    file_path,
    output_folder,
    api_url,
    api_key,
    model,
    temperature=0.3,
    top_p=0.95,
    max_tokens=2048,
    user_prompt="Convert this PDF document into structured JSON. Extract questions, their types, and options. Additionally, identify mathematical formulas and wrap them in MathML syntax enclosed within $$ delimiters.",
    system_prompt=(
        "You are a highly knowledgeable document parsing assistant. Your task is to convert the content of a PDF document containing exam questions and solutions into a structured JSON format. "
        "The PDF is divided into sections (e.g., 'Section A. Multiple choices' or 'Section B. Short-answer Questions'). Each section contains questions and their corresponding solutions. "
        "For each question, extract the following: \n"
        " - \"question\": the full text of the question (including sub-questions if any), \n"
        " - \"type\": either \"multiple-choice\" or \"short-answer\", \n"
        " - \"options\": for multiple-choice questions, a list of options; for short-answer, null. \n"
        "Also, identify all mathematical formulas, equations, and expressions. Convert each formula into valid MathML wrapped within $$ delimiters. "
        "Return only valid JSON objects corresponding to each question, following this structure:\n"
        "{\n  \"question\": \"The full text of the question\",\n  \"type\": \"multiple-choice\" or \"short-answer\",\n  \"options\": [\"Option A\", \"Option B\", ...]  \n}\n"
        "Ensure your output is JSON-parseable with no additional commentary."
    )
):
    # Validate file
    if not os.path.isfile(file_path):
        print(json.dumps({"status": "error", "message": "Input file does not exist"}))
        sys.exit(1)

    # Create output folder if it does not exist.
    os.makedirs(output_folder, exist_ok=True)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Initialize our OpenAI-compatible client with a custom base URL.
        client = openai.OpenAI(base_url=api_url, api_key=api_key)

        # Build messages using system and user prompts.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\n{file_content}"}
        ]

        # Call the ChatCompletion API (non-streaming).
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content.strip()

        # Attempt to parse the answer as JSON.
        try:
            result = json.loads(answer)
        except Exception as e:
            result = {"error": "Could not parse API response as JSON", "raw_output": answer}

        # Save the output
        output_path = os.path.join(output_folder, "converted.json")
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(result, out, indent=4)

        print(json.dumps({"status": "success", "output_file": output_path}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

# CLI entry point
if __name__ == '__main__':


    file_path = "../dataset/23_fina_1310.json"
    output_folder = "../dataset/jsonfolder"
    api_url = "https://api.siliconflow.cn/v1"
    api_key = "sk-ufvfjzrydqzznjfnqabneayuhyimirhnwekmiemjyskvxedo"

    convert_to_json_with_openai(
        file_path=file_path,
        output_folder=output_folder,
        api_url=api_url,
        api_key=api_key,
        model="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
        temperature=0.3,
        top_p=0.95,
        max_tokens=2048,
        user_prompt="Convert this PDF document into structured JSON with labeled sections and wrap mathematical formulas in MathML within $$ delimiters.",
        system_prompt=(
            "You are a highly knowledgeable document parsing assistant. Your task is to convert the content of a PDF document containing exam questions and solutions into a structured JSON format. "
            "The PDF is divided into sections (for example, 'Section A. Multiple choices' or 'Section B. Short-answer Questions'). Each section contains questions and solutions. "
            "For each question, extract:\n"
            "  - \"question\": the full text of the question (including sub-questions like part a, part b, etc.)\n"
            "  - \"type\": either \"multiple-choice\" or \"short-answer\"\n"
            "  - \"options\": for multiple-choice questions, a list of options; if not applicable, set to null.\n"
            "Identify all mathematical formulas, equations, and expressions. Convert each into valid MathML enclosed within $$ delimiters. "
            "Return only valid JSON objects for each question, using the following structure:\n"
            "{\n  \"question\": \"The full text of the question\",\n  \"type\": \"multiple-choice\" or \"short-answer\",\n  \"options\": [\"Option A\", \"Option B\", ...] \n}\n"
            "Ensure that your output is strict JSON and does not include any additional commentary."
        )
    )
