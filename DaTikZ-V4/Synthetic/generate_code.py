import json

from APIs.llms_api import GptApi
from Synthetic.prompts import construct_prompt_tikz_code
from utils import clean_tikz_code


SAVE_EVERY_N = 10


def extract_tikz(raw_output):
    first_backslash = raw_output.find('\\')
    last_backslash = raw_output.rfind('\\')
    end_of_line = raw_output.find('\n', last_backslash)
    code = raw_output[first_backslash:end_of_line]
    return code.strip()


if __name__ == "__main__":
    difficulty = "hard" # easy or medium_easy or medium_hard or hard
    input_output_file = f"datikz/loaders/synthetic/queries_{difficulty}.json"
    system_prompt_queries = "You are an expert LaTeX and TikZ programmer."
    model_tikz = GptApi(system_prompt_queries, model_id="gpt-4o", temperature=0.2, top_p=0.1)
    try:
        with open(input_output_file, "r") as f:
            all_queries = json.load(f)
    except json.JSONDecodeError:
        exit(1)
    modified = False
    for i, query in enumerate(all_queries):
        if "code" in query:
            continue
        prompt = construct_prompt_tikz_code(query)
        response = model_tikz.request(prompt)
        tikz_code = extract_tikz(response)
        cleaned_tikz_code = clean_tikz_code(tikz_code)
        new_query = {}
        for k, v in query.items():
            new_query[k] = v
            if k == "query":
                new_query["code"] = cleaned_tikz_code
        all_queries[i] = new_query
        modified = True
        if i % SAVE_EVERY_N == 0 and i != 0:
            with open(input_output_file, "w", encoding="utf-8") as f:
                json.dump(all_queries, f, indent=2, ensure_ascii=False)
    if modified:
        with open(input_output_file, "w", encoding="utf-8") as f:
            json.dump(all_queries, f, indent=2, ensure_ascii=False)
    else:
        pass