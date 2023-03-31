import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from peft.tuners.lora import Linear4bitLt

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing
    share_gradio: bool = False,
    load_4bit: bool = False,
    cpu_only: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # model, tokenizer = load_llama_model_4bit_low_ram(base_model, base_model + '/llama-7b-4bit.pt')
    # model = PeftModel.from_pretrained(model, lora_weights)
    # print('Fitting 4bit scales and zeros to half')
    # for n, m in model.named_modules():
    #     if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
    #         m.zeros = m.zeros.half()
    #         m.scales = m.scales.half()
    #         m.bias = m.bias.half()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if cpu_only:
        device = "cpu"

    if load_4bit:
        # Load 4bit model
        if device == "cuda":
            model, tokenizer = load_llama_model_4bit_low_ram(
                base_model,
                base_model + '/llama-7b-4bit.pt',
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights
            )
        else:
            model, tokenizer = load_llama_model_4bit_low_ram(
                base_model,
                base_model + '/llama-7b-4bit.pt',
                device_map={"": device},
                low_cpu_mem_usage=True
                )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        print('Fitting 4bit scales and zeros to half')
        for n, m in model.named_modules():
            if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
                m.zeros = m.zeros.half()
                m.scales = m.scales.half()
                m.bias = m.bias.half()
    else:
        # If we're not using 4bit, we can use the default model loading.
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.5,
        top_p=0.90,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    ).launch(server_name="0.0.0.0", share=share_gradio)
    # Old testing code follows.

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(main)
