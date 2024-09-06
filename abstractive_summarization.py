import ollama

from transformers import BartForConditionalGeneration, BartTokenizer
from utils import constants, logger_script

logger = logger_script.get_logger(constants.SUMMARIZATION_LOGGER_NAME)


class AbstractiveSummarizer:

    def __init__(self):
        model_name = "facebook/bart-large-cnn"
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        self.bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        pass

    def apply_bart(self, text):
        inputs = self.bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.bart_model.generate(inputs, max_length=500, min_length=300, length_penalty=1, num_beams=4,
                                     early_stopping=True)

        summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("in bart module: ", summary)
        # formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
        return summary

    def apply_llama(self, text):
        # Mistral 7B
        model = 'llama3:instruct'

        # Load the system prompt
        with open(
                'C:\\Users\\Chloe\\Documents\\MaastrichtLaw&Tech\\Thesis\\MscThesis\\summ_pipeline\\abstractive_methods\\summ_sys_prompt.txt',
                'r', encoding='utf-8') as file:
            sys_prompt = file.read()

        # Load the summarization prompt
        with open(
                'C:\\Users\\Chloe\\Documents\\MaastrichtLaw&Tech\\Thesis\\MscThesis\\summ_pipeline\\abstractive_methods\\summ_prompt.txt',
                'r', encoding='utf-8') as file:
            prompt = file.read()

            # Get response
        response = ollama.chat(model=model, keep_alive=0, options={'temperature': 0.0, 'seed': 42, "top_p": 0.0},
                               messages=[
                                   {
                                       'role': 'system',
                                       'content': sys_prompt
                                   },
                                   {
                                       'role': 'user',
                                       'content': prompt + text
                                   },
                               ])

        return response['message']['content']

