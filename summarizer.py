import spacy
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect

# Load sentiment + NER once
sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")  # spaCy for NER

# Model cache
cached_models = {}

# Language codes for MBART
LANGUAGE_CODES = {
    "English": "en_XX",
    "Arabic": "ar_AR",
    "French": "fr_XX",
    "German": "de_DE",
    "Spanish": "es_XX"
}



def summarize_article(text, model_name="facebook/bart-large-cnn", max_length=130, min_length=30, language=None):
    """
    Summarizes a given text using selected model and language (for MBART).
    Automatically detects language if not provided.
    """
    if len(text.strip()) == 0:
        return "No content available."

    # Detect language if not manually specified
    if language is None:
        try:
            detected = detect(text)
            language = {
                "en": "English",
                "ar": "Arabic",
                "fr": "French",
                "de": "German",
                "es": "Spanish"
            }.get(detected, "English")
        except:
            language = "English"

    if model_name == "facebook/mbart-large-cc25":
        if "mbart" not in cached_models:
            model = MBartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            cached_models["mbart"] = (model, tokenizer)
        else:
            model, tokenizer = cached_models["mbart"]

        lang_code = LANGUAGE_CODES.get(language, "en_XX")
        tokenizer.src_lang = lang_code

        inputs = tokenizer(text[:1024], return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[lang_code],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            no_repeat_ngram_size=3
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    else:
        if model_name not in cached_models:
            cached_models[model_name] = pipeline("summarization", model=model_name)
        summarizer = cached_models[model_name]

        if len(text) > 1024:
            text = text[:1024]

        input_length = len(text.split())
        adjusted_max = min(max_length, input_length * 2)
        adjusted_min = min(min_length, input_length)

        summary = summarizer(text, max_length=adjusted_max, min_length=adjusted_min, do_sample=False)
        return summary[0]['summary_text']


def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])[0]
    return result['label'], result['score']

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]

def style_entity_tags(entities):
    colors = {
        'PERSON': '#1f77b4',
        'ORG': '#2ca02c',
        'GPE': '#d62728',
        'LOC': '#9467bd',
        'PRODUCT': '#ff7f0e',
        'EVENT': '#8c564b',
        'DEFAULT': '#7f7f7f'
    }

    html_tags = []
    for text, label in entities:
        color = colors.get(label, colors['DEFAULT'])
        tag = f'<span style="background-color:{color}; color:white; padding:2px 6px; margin:2px; border-radius:5px; font-size:0.85em;">{text} ({label})</span>'
        html_tags.append(tag)

    return " ".join(html_tags)
