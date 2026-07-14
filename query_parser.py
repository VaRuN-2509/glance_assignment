"""Uses an LLM to convert a free-text query into a structured object that mirrors
Fashionpedia's own schema (category + attributes per garment, plus a scene string).
This structured form is what lets retrieval do exact/symbolic matching against each
segment's metadata, instead of relying only on one blended CLIP similarity score.

Output schema:
{
  "garments": [ {"category": "shirt", "attributes": ["red", "long sleeve"]}, ... ],
  "scene": "modern office"   # free text describing environment/context, "" if none
}
"""
import os
import json
import anthropic

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "claude-sonnet-4-5"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def format_text(model_name,prompt):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )


    messages = [
    {"role": "user", "content": parse_query(prompt)}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    print("content:", content)

    return json.loads(content)


def parse_query(query, vocab_path="chroma_db/vocab.json"):
    with open(vocab_path) as f:
        vocab = json.load(f)

    # print(f"Allowed categories: {vocab['categories']}")
    # print(f"Allowed attributes: {vocab['attributes']}")

    prompt = f"""Convert the fashion search query below into JSON with this exact shape:
{{"garments": [{{"category": "<one of the allowed categories>", "attributes": ["<from allowed attributes>", ...]}}], "scene": "<short environment/context phrase, or empty string>"}}

Allowed categories: {vocab['categories']}
Allowed attributes: {vocab['attributes']}


Rules:
- Only use categories/attributes from the allowed lists (pick the closest match).
- One entry per distinct garment mentioned in the query.
- "scene" captures only location/environment/vibe words (e.g. "modern office", "park"), not garments.
- Output ONLY the JSON object, no other text.

Query: "{query}"
"""

    return prompt


if __name__ == "__main__":
    print(format_text(model_name = "Qwen/Qwen3-8B", prompt ="A red tie and a white shirt in a formal setting."))