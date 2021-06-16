from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

import sys

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--input_file', type=str, default=None,
                        help="file")
    parser.add_argument('--output_file', type=str, default=None,
                        help="file")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, pad_token='<PAD>', padding_side='left')
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    fin = open(args.input_file)
    inputs = [i.strip() for i in fin]
    output_tests = []
    for idx in range(0, len(inputs), args.batch_size):
        logger.info(f"PROGRESS: {int(idx/len(inputs)*100)}%")

        # raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        raw_inputs = []
        for i in range(idx, min(idx+args.batch_size, len(inputs))):
            lines = inputs[i]
            raw_text = lines.split(' & ')[0] + ' & '
            if args.model_type in ["transfo-xl", "xlnet"]:
                # Models with memory likes to have a long prompt for short inputs.
                raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
            raw_inputs.append(raw_text)
        
        encoding_inputs = tokenizer.batch_encode_plus(raw_inputs, pad_to_max_length=True, add_special_tokens=False)
        context_tokens = torch.LongTensor(encoding_inputs['input_ids']).to(args.device)
        max_length = len(context_tokens[0])
        attention_mask = torch.LongTensor(encoding_inputs['attention_mask']).to(args.device)
        position_ids = (attention_mask.cumsum(-1) - 1)
        position_ids.masked_fill_(attention_mask==0, 0)

        if args.model_type == "ctrl":
            if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
        out_ids = model.generate(
            input_ids=context_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_beams=args.num_samples,
            num_return_sequences=args.num_samples,
            max_length=args.length,
            temperature=args.temperature,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        out_ids = out_ids.reshape(len(raw_inputs), args.num_samples, -1)[:, :, max_length:].tolist()
        for j, out in enumerate(out_ids):
            examples = [inputs[j]]
            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = text[: text.find(args.stop_token) if args.stop_token else None]
                examples.append(text)
            output_tests.append(examples)
        # break
        # if args.prompt:
            # break
    import json
    json.dump(output_tests, open(args.output_file,'w'), indent=2)
    return text

if __name__ == '__main__':
    main()
