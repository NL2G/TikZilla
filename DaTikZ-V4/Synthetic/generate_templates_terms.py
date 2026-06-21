import os
import json
import difflib

from APIs.llms_api import GptApi
from Synthetic.prompts import construct_prompt_template_terms_easy, construct_prompt_template_terms_medium_easy, construct_prompt_template_terms_medium_hard, construct_prompt_template_terms_hard


SIMILARITY_THRESHOLD = 0.95
TEMPLATES_PER_QUERY = 5
NUM_ITERATIONS = 100


def is_similar(new_template, existing_templates):
    for template in existing_templates:
        similarity = difflib.SequenceMatcher(None, new_template, template).ratio()
        if similarity > SIMILARITY_THRESHOLD:
            return True
    return False


if __name__ == "__main__":
    difficulty = "easy" # easy or medium_easy or medium_hard or hard
    output_file = f"datikz/loaders/synthetic/templates_terms_{difficulty}.json"
    system_prompt_templates_terms = "You are an expert in generating diverse and high quality templates and terms for scientific captions."
    model_template_terms = GptApi(system_prompt_templates_terms, model_id="gpt-4o", temperature=0.7, top_p=0.8)
    if difficulty == "easy":
        prompt = construct_prompt_template_terms_easy(TEMPLATES_PER_QUERY)
    elif difficulty == "medium_easy":
        prompt = construct_prompt_template_terms_medium_easy(TEMPLATES_PER_QUERY)
    elif difficulty == "medium_hard":
        prompt = construct_prompt_template_terms_medium_hard(TEMPLATES_PER_QUERY)
    elif difficulty == "hard":
        prompt = construct_prompt_template_terms_hard(TEMPLATES_PER_QUERY)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, "r") as f:
                all_templates_terms = json.load(f)
        except json.JSONDecodeError:
            all_templates_terms = {}
    else:
        all_templates_terms = {}
    existing_template_texts = [
        entry["template"] for entry in all_templates_terms.values()
        if isinstance(entry, dict) and "template" in entry
    ]
    template_counter = len(all_templates_terms)
    for i in range(NUM_ITERATIONS):
        response = model_template_terms.request(prompt)
        response_str = response.strip("```json").strip("```").strip()
        try:
            raw_data = json.loads(response_str)
        except json.JSONDecodeError:
            continue
        for j in range(TEMPLATES_PER_QUERY):
            template_key = f"template_{j + 1}"
            terms_key = f"terms_{j + 1}"
            if template_key not in raw_data:
                continue
            template_text = raw_data[template_key]
            terms_dict = raw_data.get(terms_key, {})
            if is_similar(template_text, existing_template_texts):
                continue
            template_counter += 1
            new_key = f"template_{template_counter}"
            all_templates_terms[new_key] = {
                "template": template_text,
                "terms": terms_dict
            }
            existing_template_texts.append(template_text)
            if len(template_counter) >= TEMPLATES_PER_QUERY:
                break
        with open(output_file, "w") as f:
            json.dump(all_templates_terms, f, indent=2, ensure_ascii=False)