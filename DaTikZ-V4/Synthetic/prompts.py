def construct_prompt_template_terms_easy(num_templates):
    structure = """
    Your task is to generate a set of simple templates and corresponding terms that will be used to describe scientific or technical images in a way that is easy to convert into TikZ code.

    ### Requirements:
    1. The templates should be **simple**, **concrete**, and focus on **basic visual elements** like lines, arrows, points, shapes, colors, and spatial relationships.
    2. Avoid mathematical expressions, nested phrases, or abstract descriptions.
    3. Each template should use clearly drawable elements — for example: a red circle above a blue square.
    4. Use the syntax `<term>` (e.g., `<color>`, `<shape>`, `<direction>`, `<number>`, etc.) in your templates.
    5. For each `<term>`, provide a list of possible values **only** if the term has a small, finite set of common options (e.g., `<shape>`, `<color>`, `<direction>`, etc.).
    6. Format the output as valid JSON with correct structure and syntax.
    7. Generate at least {num_templates} unique templates.

    ### Example Output (Valid JSON):
    {{
    "template_1": "A <texture> <shape> at the <position>.",
    "terms_1": {{
        "texture": ["hatched", "striped", "solid", "dotted"],
        "shape": ["circle", "square", "triangle", "rectangle"],
        "position": ["center", "top", "bottom", "left", "right"]
        }},
    "template_2": "A <shape> connected to another <shape> with a <line_style> line.",
    "terms_2": {{
        "shape": ["circle", "square", "ellipse"],
        "line_style": ["solid", "dashed", "dotted"]
        }},
    "template_3": "An arrow pointing <direction> from a <color> <shape>.",
    "terms_3": {{
        "direction": ["up", "down", "left", "right"],
        "color": ["red", "black", "blue"],
        "shape": ["rectangle", "circle", "square"]
        }},
    "template_4": "Two <shapes> placed side by side.",
    "terms_4": {{
        "shapes": ["circles", "squares", "triangles"]
        }},
    "template_5": "A <label> above a <color> <shape>.",
    "terms_5": {{
        "label": ["A", "B", "X", "Y", "Z"],
        "color": ["orange", "black", "pink"],
        "shape": ["circle", "square", "ellipse"]
        }}
    }}

    Your JSON Output:
    """
    return structure.format(num_templates=num_templates)


def construct_prompt_template_terms_medium_easy(num_templates):
    structure = """
    Your task is to generate a set of highly regular and structured templates with corresponding terms. These templates will be used to describe scientific or technical diagrams composed of repeatable geometric or labeled patterns (e.g., layers, grids, arrows, labeled blocks), which will later be converted into TikZ code.

    ### Requirements:
    1. Keep templates regular and syntactically simple.
    2. Emphasize repeatable, layout-friendly structures like chains, stacks, layers, sequences, grids, or labeled boxes.
    3. Use terms like `<block_type>`, `<direction>`, `<label>`, `<position>`, `<connection_type>` that align well with TikZ-style drawing.
    4. Avoid unnecessary complexity — the focus is on **structure**, not domain richness.
    5. Use `<term>` syntax and include lists only for bounded terms like `<direction>`, `<block_type>`, `<label>`.
    6. Generate at least {num_templates} templates and output valid JSON.

    ### Example Output (Valid JSON):
    {{
    "template_1": "A row of <number> <block_type> blocks labeled <label_sequence>.",
    "terms_1": {{
        "block_type": ["neuron", "module", "unit", "gate", "cell"],
        "label_sequence": ["A, B, C", "1, 2, 3", "X, Y, Z"]
    }},
    "template_2": "A <block_type> connected to another <block_type> with a <connection_type> arrow.",
    "terms_2": {{
        "block_type": ["layer", "node", "unit"],
        "connection_type": ["solid", "dashed", "curved"]
    }},
    "template_3": "A grid of <number> <shape> elements aligned <direction>.",
    "terms_3": {{
        "shape": ["square", "circle", "rectangle"],
        "direction": ["left-to-right", "top-down"]
    }},
    "template_4": "Each <shape> has a label <label> on top.",
    "terms_4": {{
        "shape": ["circle", "box"],
        "label": ["A", "B", "C", "X", "Y", "Z"]
    }},
    "template_5": "Three <block_type> blocks connected in sequence from <direction>.",
    "terms_5": {{
        "block_type": ["layer", "unit", "stage"],
        "direction": ["left", "top"]
        }}
    }}

    Your JSON Output:
    """
    return structure.format(num_templates=num_templates)


def construct_prompt_template_terms_medium_hard(num_templates):
    structure = """
    Your task is to generate a set of structured templates and corresponding terms that describe **scientific or technical diagrams** in a way that is easy to convert into TikZ code.

    ### Goals:
    These templates should represent **scientific structures** such as flowcharts, layered models, labeled networks, modular systems, or stepwise procedures. Each description should be **visually structured and layout-friendly**, allowing TikZ-based diagrams to be generated from them reliably.

    ### Requirements:
    1. Focus on **scientific diagrams**, not artistic or purely abstract visuals.
    2. Each template must describe a layout involving **clearly defined components** like blocks, arrows, stages, units, layers, or labeled nodes.
    3. Emphasize **logical or spatial organization** (e.g., left-to-right, top-down, in a grid, as a sequence).
    4. Use placeholder syntax like `<term>` (e.g., `<label>`, `<block_type>`, `<direction>`, `<step>`).
    5. Provide term lists only for bounded options (like directions, labels, shapes, block types).
    6. Generate **at least {num_templates} templates** and format the output as valid JSON.

    ### Example Output (Valid JSON):
    {{
        "template_1": "A <number>-step process from <start_label> to <end_label> using <block_type> blocks aligned <direction>.",
        "terms_1": {{
            "block_type": ["analysis", "filter", "transformation", "gate"],
            "direction": ["left-to-right", "top-down"],
            "start_label": ["Input", "Sensor", "Data"],
            "end_label": ["Output", "Result", "Classifier"]
        }},
        "template_2": "A <layer_type> layer connected to a <layer_type> layer with a <line_style> arrow.",
        "terms_2": {{
            "layer_type": ["convolution", "dense", "activation", "pooling"],
            "line_style": ["solid", "dashed", "curved"]
        }},
        "template_3": "Three <color> <shape> nodes labeled <label_set> placed in a triangle.",
        "terms_3": {
            "color": ["blue", "green", "orange"],
            "shape": ["circle", "square"],
            "label_set": ["A, B, C", "X, Y, Z", "1, 2, 3"]
        }},
        "template_4": "A grid of <shape> elements showing different <measurement_type> values.",
        "terms_4": {{
            "shape": ["rectangles", "squares", "cells"],
            "measurement_type": ["temperature", "voltage", "intensity"]
        }},
        "template_5": "Each <shape> represents a <scientific_element> and is linked with a <connection_type> line.",
        "terms_5": {{
            "shape": ["circle", "hexagon", "box"],
            "scientific_element": ["gene", "sensor", "device", "neuron"],
            "connection_type": ["dashed", "solid", "arrow"]
        }}
    }}

    Your JSON Output:
    """
    return structure.format(num_templates=num_templates)


def construct_prompt_template_terms_hard(num_templates):
    structure = """
    Your task is to generate a set of templates and corresponding terms that can later be used to form a set of varying descriptions of scientific images.

    ### Requirements:
    1. The templates should be highly generalizable so that many different terms can be used to fill in one template and generate diverse descriptions.
    2. Cover a wide range of scientific fields such as computer science, mathematics, physics, electronics, mechanics, biology, and more.
    3. Mark terms in each template using the syntax `<term>` (e.g., `<number>`, `<color>`, `<shape>`, etc.).
    4. For each `<term>`, provide a corresponding list of possible values **only** if it has a relatively small, finite set of options (e.g., `<color>`, `<shape>`, `<texture>`, `<chart_type>`). Do **not** include lists for open-ended values like `<number>`, `<title>`, `<label>`, etc.
    5. Avoid duplicate entries within the lists.
    6. Format the response as valid JSON, ensuring proper use of commas, quotation marks, spacing, and nesting.

    ### Output Requirements:
    - Generate at least {num_templates} unique templates.
    - Use fields relevant to scientific diagrams, graphs, data plots, schematics, etc.
    - Use the following format:
    - Each template key should be named as `"template_1"`, `"template_2"`, etc.
    - Each corresponding term list should be named `"terms_1"`, `"terms_2"`, etc.

    ### Example Output with three templates (Valid JSON):
    {{
    "template_1": "A <math_function> (<color> <texture> texture) with <property> at <number>.",
    "terms_1": {{
        "math_function": ["Exponential", "Rational", "Linear", "Polynomial", "Quadratic", "Cosine", "Trigonometric", "Parabola"],
        "color": ["green", "yellow", "blue", "magenta", "orange", "brown"],
        "texture": ["dotted", "striped", "dashed", "solid"],
        "property": ["opening", "global maximum", "local minimum", "intersection", "zero", "turning point", "symmetry"]
        }},
    "template_2": "A <shape> <spatial_relation> a <shape_2> and <shape_2> <spatial_relation_2> <shape_3>.",
    "terms_2": {{
        "shape": ["arrow", "line", "circle", "square", "ellipse", "point", "rectangle"],
        "spatial_relation": ["left", "between", "inside", "top right", "below"]
        }},
    "template_3": "A <chart> with <number> <data> and legend <spatial_relation> titled <title>.",
    "terms_3": {{
        "chart": ["bar", "pie", "line", "scatter", "pyramid", "funnel", "histogram", "radar", "bubble", "waterfall", "candlestick", "arc"],
        "data": ["bars", "points", "lines", "marks", "clusters", "segments", "layers", "outliers"],
        "spatial_relation": ["top right", "bottom", "middle", "outside", "inside"]
        }},
    "template_4": "A <molecule_type> diagram with <bond_type> bonds between <element> atoms in a <structure> configuration.",
    "terms_4": {{
        "molecule_type": ["organic", "inorganic", "cyclic", "aromatic", "linear"],
        "bond_type": ["single", "double", "triple", "hydrogen"],
        "element": ["carbon", "oxygen", "nitrogen", "hydrogen", "sulfur"],
        "structure": ["ring", "chain", "branched", "tetrahedral"]
        }},
    "template_5": "A schematic showing <component> connected in a <topology> circuit with <measurement> indicators.",
    "terms_5": {{
        "component": ["resistor", "capacitor", "inductor", "diode", "transistor"],
        "topology": ["series", "parallel", "mesh", "bridge"],
        "measurement": ["voltage", "current", "resistance", "frequency"]
        }}
    }}

    Your JSON Output:
    """
    return structure.format(num_templates=num_templates)


def construct_prompt_fill_queries(template, terms, num_queries):
    structure = """
    Your task is to fill in the brackets `<...>` in the provided template using the provided lists of terms below.
    Each filled-in version should result in a **scientifically meaningful and visually representable description**.

    ### Requirements:
    1. Choose terms that logically and visually align with each other.
    2. If a bracket has no associated term list, generate a fitting scientific term yourself.
    3. Generate exactly **{num_queries}** unique and high-quality descriptions.
    4. Return them as **valid JSON**, using keys like `"description_1"`, `"description_2"`, ..., `"description_{num_queries}"`.

    Your Template:
    {template}

    Available Term Lists:
    {terms}

    Your JSON Output:
    """
    return structure.format(template=template, terms=terms, num_queries=num_queries)


def construct_prompt_tikz_code(query):
    structure = """
    Your task is to generate complete, and scientifically accurate TikZ figures based on the provided description.

    ### Requirements:
    1. Output only valid LaTeX code.
    2. Include the full document, including preamble and document environment.
    3. Do **not** include any explanatory text or comments, only the LaTeX code.
    4. Use the document class `\\documentclass[tikz]{{standalone}}` and wrap all TikZ code inside `\\begin{{document}}` ... `\\end{{document}}`.
    5. Do **not** create any external data files.

    Description:
    {query}

    Your TikZ code:
    """
    return structure.format(query=query)