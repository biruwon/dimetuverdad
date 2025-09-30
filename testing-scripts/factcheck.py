from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend
from openai_harmony import (
    SystemContent,
    Message,
    Role,
    Conversation,
    load_harmony_encoding,
    HarmonyEncodingName
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------
# Load GPT-OSS-20B
# -----------------------
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# -----------------------
# Enable Browser Tool
# -----------------------
system_message_content = (
    SystemContent.new()
    .with_conversation_start_date("2025-08-29")
    .with_browser_tool()  # enables search/open/find
)
system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)

# -----------------------
# Build Conversation
# -----------------------
conversation = Conversation(
    messages=[
        system_message,
        Message.from_role_and_content(
            Role.USER,
            "Fact-check this post: 'Spain is leaving the European Union in 2025'."
        )
    ]
)

# -----------------------
# Run Inference
# -----------------------
inputs = tokenizer.apply_chat_template(
    load_harmony_encoding(HarmonyEncodingName.LATEST).encode(conversation),
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.2,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
