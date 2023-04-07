from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

# Шаблоны для проверки качества текста
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Контекстная информация приведена ниже. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Учитывая контекстную информацию, ответьте на следующий вопрос "
    "(если вы не знаете ответа, используйте лучшее из того, что вам известно): {query_str}\n"
)
TEXT_QA_TEMPLATE = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)

# Refine templates
DEFAULT_REFINE_PROMPT_TMPL = (
    "Первоначальный вопрос звучит следующим образом: {query_str}\n"
    "Мы предоставили существующий ответ: {existing_answer}\n"
    "У нас есть возможность уточнить существующий ответ "
    "(только при необходимости) с некоторым дополнительным контекстом ниже.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Учитывая новый контекст и используя все свои знания, улучшите существующий ответ. "
    "Если вы не можете улучшить существующий ответ, просто повторите его снова. "
    "Не упоминайте, что вы прочитали вышеупомянутый контекст."
)
DEFAULT_REFINE_PROMPT = RefinePrompt(DEFAULT_REFINE_PROMPT_TMPL)

CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "У нас есть возможность уточнить вышеприведенный ответ "
        "(только при необходимости) с некоторым дополнительным контекстом ниже.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Учитывая новый контекст и используя все свои знания, улучшите существующий ответ. "
        "Если вы не можете улучшить существующий ответ, просто повторите его снова. "
        "Не упоминайте, что вы прочитали вышеупомянутый контекст."
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

# уточнить селектор подсказок
DEFAULT_REFINE_PROMPT_SEL_LC = ConditionalPromptSelector(
    default_prompt=DEFAULT_REFINE_PROMPT.get_langchain_prompt(),
    conditionals=[(is_chat_model, CHAT_REFINE_PROMPT.get_langchain_prompt())],
)
REFINE_TEMPLATE = RefinePrompt(
    langchain_prompt_selector=DEFAULT_REFINE_PROMPT_SEL_LC
)

DEFAULT_TERM_STR = (
)

DEFAULT_TERMS = {
}
