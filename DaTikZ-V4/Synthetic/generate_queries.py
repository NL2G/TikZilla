import os
import json
import difflib

from APIs.llms_api import GptApi
from Synthetic.prompts import construct_prompt_fill_queries


SIMILARITY_THRESHOLD = 0.95
QUERIES_PER_TEMPLATE = 10


def is_similar(new_template, existing_templates):
    for template in existing_templates:
        similarity = difflib.SequenceMatcher(None, new_template, template).ratio()
        if similarity > SIMILARITY_THRESHOLD:
            return True
    return False


def save_queries(output_file, queries):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.extend(queries)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    difficulty = "hard" # easy or medium_easy or medium_hard or hard
    input_file = f"datikz/loaders/synthetic/templates_terms_{difficulty}.json"
    output_file = f"datikz/loaders/synthetic/queries_{difficulty}.json"
    system_prompt_queries = "You are an expert in generating diverse and high quality scientific descriptions based on templates and terms."
    model_template_terms = GptApi(system_prompt_queries, model_id="gpt-4o", temperature=0.6, top_p=0.7)
    try:
        with open(input_file, "r") as f:
            all_templates_terms = json.load(f)
    except json.JSONDecodeError:
        exit(1)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                saved_queries = json.load(f)
            except json.JSONDecodeError:
                saved_queries = []
    else:
        saved_queries = []
    existing_queries_texts = [q["query"] for q in saved_queries]
    new_queries_to_save = []
    for idx, (template_id, item) in enumerate(all_templates_terms.items()):
        current_template = item["template"]
        current_terms = item["terms"]
        prompt = construct_prompt_fill_queries(current_template, current_terms, QUERIES_PER_TEMPLATE)
        response = model_template_terms.request(prompt)
        response_str = response.strip("```json").strip("```").strip()
        try:
            raw_data = json.loads(response_str)
        except json.JSONDecodeError:
            continue
        generated_for_this_template = []
        for j, (key, gen_query) in enumerate(raw_data.items()):
            if is_similar(gen_query, existing_queries_texts + [q["query"] for q in generated_for_this_template]):
                continue
            query_entry = {
                "query": gen_query,
                "query_id": f"synthetic_{difficulty}_{idx * QUERIES_PER_TEMPLATE + j}"
            }
            generated_for_this_template.append(query_entry)
            if len(generated_for_this_template) >= QUERIES_PER_TEMPLATE:
                break
        if generated_for_this_template:
            new_queries_to_save.extend(generated_for_this_template)
            save_queries(output_file, new_queries_to_save)
            existing_queries_texts.extend([q["query"] for q in generated_for_this_template])
            new_queries_to_save = []