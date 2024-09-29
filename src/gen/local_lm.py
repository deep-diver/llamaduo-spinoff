import torch

from peft import PeftModel, PeftConfig

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from alignment.model_utils import (
    get_tokenizer,
    get_quantization_config,
    get_kbit_device_map
)

def get_model(model_id, model_revision, model_args, data_args, sft_args):
    """
    get_model instantiates and return fine-tuned language model and tokenzier.

    arguments:
    model_args -- ModelArguments obtained from H4ArgumentParser
    data_args -- DataArguments obtained from H4ArgumentParser
    """
    model_id = sft_args.hub_model_id if model_id is None else model_id
    # tokenizer = get_tokenizer(model_args, data_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )    
    quantization_config = get_quantization_config(model_args)
    print(quantization_config)
    print(model_id)
    model_kwargs = dict(
        revision=model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )    
    print(model_args.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_id)
    model = model.merge_and_unload()
    #model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return tokenizer, model_id, model

def gen_model_outputs(model, tokenizer, batch_data, ft_model_gen_configs, delimiter):
    """
    gen_model_output generates and return response(output) from a given model.

    arguments:
    model -- fine-tuned lanaguage model instance
    tokenizer -- tokenizer instance
    ds -- a batch data records which has "prompt" column
    """
    input_ids = tokenizer(
        batch_data["input_ids"], 
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(
        **input_ids,
        **ft_model_gen_configs,
    )
   
    outputs = []
    raw_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for idx, raw_output in enumerate(raw_outputs):
        i_data = batch_data['prompt'][idx]
        i_data = raw_output.split(i_data)
        try:
            output = i_data[1].split(delimiter)[1].strip()
        except:
            print(i_data)
        outputs.append(output)

    return outputs
