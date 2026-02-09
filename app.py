import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re

# Global variables to store model and tokenizer
model = None
tokenizer = None
MODEL_NAME = "shukdevdattaEX/DeepSeek-R1-Bengali-Hate-Speech-Classification-Multi-Class-merged"

# Define the prompt template (same as used in training)
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the text and analyze it step by step to determine the correct classification.
### Instruction:
You are an expert in Bengali text analysis and classification. Your task is to classify Bengali text into one of the following categories:
- Abusive: Text containing abusive or offensive language
- Sexism: Text containing sexist content or discriminatory language against gender
- Religious Hate: Text containing hate speech targeting religious groups
- Political Hate: Text containing hate speech targeting political groups or individuals
- Profane: Text containing profane or vulgar language
- None: Text that doesn't fall into any of the above categories
### Text:
{}
### Classification:
<think>"""

def load_model(hf_token):
    """Load the model with the provided HF token"""
    global model, tokenizer
    
    if not hf_token or not hf_token.strip():
        return "❌ Please enter a valid Hugging Face token!", gr.update(visible=False)
    
    try:
        # Authenticate
        print("🔐 Authenticating with Hugging Face...")
        login(hf_token.strip())
        print("✅ Authentication successful!")
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=hf_token.strip(),
            trust_remote_code=True
        )
        
        # Load model
        print("🚀 Loading model (this may take a few minutes on CPU)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=hf_token.strip(),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        
        return "✅ Model loaded successfully! You can now classify Bengali text.", gr.update(visible=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"❌ Error loading model: {str(e)}\n\nPlease check your token and try again.", gr.update(visible=False)

def classify_bengali_text(text, temperature=0.1, max_tokens=300):
    """
    Classify Bengali text using the fine-tuned model
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return """
        <div style="padding: 15px; border-radius: 10px; background: #FFE4E4; border: 2px solid #FF6B6B;">
            <h3 style="color: #C92A2A;">❌ Model Not Loaded</h3>
            <p>Please load the model first by entering your Hugging Face token above.</p>
        </div>
        """
    
    if not text.strip():
        return """
        <div style="padding: 15px; border-radius: 10px; background: #FFF4E6; border: 2px solid #FFA94D;">
            <h3 style="color: #D9480F;">⚠️ No Input</h3>
            <p>Please enter some Bengali text to classify.</p>
        </div>
        """

    try:
        # Format the prompt
        formatted_prompt = prompt_style.format(text.strip())

        # Tokenize the input
        inputs = tokenizer(
            [formatted_prompt], 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move inputs to CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        # Generate response
        print("🤔 Generating classification...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the response
        full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract the classification part
        if "### Classification:" in full_response:
            classification_part = full_response.split("### Classification:")[1]
        else:
            classification_part = full_response

        # Extract reasoning and final classification
        reasoning = ""
        final_classification = ""

        if "<think>" in classification_part:
            if "</think>" in classification_part:
                think_pattern = r'<think>(.*?)</think>'
                think_match = re.search(think_pattern, classification_part, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    final_classification = classification_part.split("</think>")[-1].strip()
                else:
                    reasoning = classification_part.split("<think>")[-1].strip()
            else:
                reasoning = classification_part.split("<think>")[-1].strip()
        else:
            final_classification = classification_part.strip()

        # Clean up the classification result
        if not final_classification and reasoning:
            lines = reasoning.split('\n')
            for line in reversed(lines):
                if any(category in line for category in ['Abusive', 'Sexism', 'Religious Hate', 'Political Hate', 'Profane', 'None']):
                    final_classification = line.strip()
                    break

        if not final_classification:
            lines = classification_part.strip().split('\n')
            final_classification = lines[-1].strip() if lines else "Unable to classify"

        # Format output
        result_html = f"""
        <div style="padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #2E8B57; margin-bottom: 10px;">🎯 Classification Result:</h3>
            <p style="font-size: 18px; font-weight: bold; color: #1E4B87; background: #E6F3FF; padding: 10px; border-radius: 5px; margin: 5px 0;">
                {final_classification}
            </p>
            <h4 style="color: #B8860B; margin: 15px 0 10px 0;">🧠 Model Reasoning:</h4>
            <p style="background: #FFF8DC; padding: 10px; border-radius: 5px; border-left: 4px solid #DAA520; font-style: italic;">
                {reasoning if reasoning else "No detailed reasoning provided."}
            </p>
        </div>
        """
        
        return result_html

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return f"""
        <div style="padding: 15px; border-radius: 10px; background: #FFE4E4; border: 2px solid #FF6B6B;">
            <h3 style="color: #C92A2A;">❌ Error</h3>
            <p>{str(e)}</p>
        </div>
        """

# Define example texts for testing
example_texts = [
    ("আজকের আবহাওয়া খুব সুন্দর", "None - Neutral"),
    ("আমি আজ বাজারে গিয়েছিলাম এবং অনেক মজার জিনিস দেখেছি", "None - Neutral"),
    ("তুমি একেবারে বোকা এবং নির্বোধ মানুষ", "Abusive"),
    ("এই লোকটা পুরোপুরি অযোগ্য এবং বেকুব", "Abusive"),
    ("মেয়েদের ঘরে থাকা উচিত, তারা বাইরের কাজের জন্য উপযুক্ত নয়", "Sexism"),
    ("নারীরা গাড়ি চালাতে পারে না, তারা শুধু রান্না করতে পারে", "Sexism"),
    ("এই ধর্মের সব মানুষ খারাপ এবং সমাজের জন্য ক্ষতিকর", "Religious Hate"),
    ("ওই ধর্মাবলম্বীরা আমাদের দেশে থাকার যোগ্য নয়", "Religious Hate"),
    ("এই দলের সব নেতা চোর এবং দুর্নীতিবাজ", "Political Hate"),
    ("এই রাজনীতিবিদ দেশের শত্রু", "Political Hate"),
    ("তোমার মুখ বন্ধ কর, নাহলে খারাপ পরিণতি হবে", "Profane"),
    ("এই জিনিসটা একদম বাজে এবং নিকৃষ্ট মানের", "Profane"),
]

# Create Gradio interface
with gr.Blocks(
    title="Bengali Hate Speech Classification",
    theme=gr.themes.Ocean(),
    css="""
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    """
) as iface:

    gr.HTML("""
    <div class="main-header">
        🇧🇩 Bengali Hate Speech Classification
    </div>
    <p style="text-align: center; font-size: 1.2em; color: #555; margin-bottom: 30px;">
        Classify Bengali text into categories: Abusive, Sexism, Religious Hate, Political Hate, Profane, or None
    </p>
    """)

    # Model Loading Section
    with gr.Group():
        gr.Markdown("### 🔑 Step 1: Load Model")
        gr.Markdown("Enter your Hugging Face token to load the model. You can get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")
        
        with gr.Row():
            token_input = gr.Textbox(
                label="Hugging Face Token",
                placeholder="hf_...",
                type="password",
                scale=3
            )
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
        
        load_status = gr.Markdown("")

    # Classification Section (initially hidden)
    with gr.Group(visible=False) as classification_section:
        gr.Markdown("### 📝 Step 2: Classify Text")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="🇧🇩 Enter Bengali Text",
                    placeholder="এখানে বাংলা টেক্সট লিখুন...",
                    lines=4,
                    max_lines=8
                )

                with gr.Row():
                    classify_btn = gr.Button("🔍 Classify Text", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.1,
                        step=0.01,
                        label="Temperature (Lower = More Consistent)",
                        info="Controls randomness in classification"
                    )

                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=300,
                        step=10,
                        label="Max Tokens",
                        info="Maximum length of generated response"
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Example Texts")
                gr.Markdown("Click on any example to test:")
                
                for example_text, category in example_texts:
                    gr.Button(
                        f"{category[:20]}...",
                        size="sm",
                        variant="outline"
                    ).click(
                        lambda x=example_text: x,
                        outputs=text_input
                    )

        result_output = gr.HTML(label="Classification Result")

    # Event handlers
    load_btn.click(
        fn=load_model,
        inputs=[token_input],
        outputs=[load_status, classification_section]
    )

    classify_btn.click(
        fn=classify_bengali_text,
        inputs=[text_input, temperature, max_tokens],
        outputs=result_output
    )

    clear_btn.click(
        lambda: ("", ""),
        outputs=[text_input, result_output]
    )

    # Add footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: #F0F8FF; border-radius: 10px;">
        <p style="color: #555;">
            🤖 Powered by <strong>DeepSeek-R1</strong> fine-tuned for Bengali text classification<br>
            📊 Model: <em>shukdevdattaEX/DeepSeek-R1-Bengali-Hate-Speech-Classification-Multi-Class</em><br>
            💻 Running on CPU
        </p>
        <p style="color: #888; font-size: 0.9em; margin-top: 10px;">
            ⚠️ Note: Classification may take longer on CPU. First-time model loading may take several minutes.
        </p>
    </div>
    """)

print("🎉 Gradio app is ready!")
print("🚀 Starting the app...")

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)
