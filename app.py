import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import time

st.set_page_config(page_title="Legal LLaMA DPO", page_icon="⚖️", layout="wide")

st.markdown("<h1 style='text-align:center;'>⚖️ Legal AI Chatbot (DPO Fine-Tuned)</h1>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    base_model_id = "meta-llama/Llama-3.2-1B"
    dpo_model_id = "models/Llama-3.2-1B-DPO"  

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in float16 for memory efficiency and compatibility
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    try:
        # Load PEFT model and merge
        model = PeftModel.from_pretrained(base_model, dpo_model_id)
        model = model.merge_and_unload()
    except Exception as e:
        st.error(f"Error merging LoRA weights: {e}")
        # Fallback to base model if merging fails
        model = base_model

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.6,
        do_sample=True,
    )

    return pipe, tokenizer


pipe, tokenizer = load_model()

def ask(question):
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{question}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    response = pipe(prompt)[0]["generated_text"]
    return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()


st.subheader("💬 Enter your legal question")
user_input = st.text_area("")

if st.button("Generate Response"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            start = time.time()
            answer = ask(user_input)
            end = time.time()

        st.markdown("### ✅ Response:")
        st.write(answer)
        st.markdown(f"🕒 _Time taken: {end - start:.2f} seconds_")
