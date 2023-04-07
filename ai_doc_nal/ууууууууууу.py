import os
import streamlit as st
import PyPDF2
from PIL import Image
from llama_index import (
    Document,
    GPTSimpleVectorIndex,
    GPTListIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    PromptHelper,
)
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR, ImageParser

from constants import DEFAULT_TERM_STR, DEFAULT_TERMS, REFINE_TEMPLATE, TEXT_QA_TEMPLATE
from utils import get_llm


if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS

# Декоратор для кэширования ресурсов.
@st.cache_resource
def get_file_extractor():
    # Создание парсера изображений для файлов .jpg, .png, .jpeg, и .pdf.
    image_parser = ImageParser(keep_image=True, parse_text=True)
    # Инициализация парсера файлов по умолчанию.
    file_extractor = DEFAULT_FILE_EXTRACTOR
    # Обновление парсера, чтобы добавить обработку новых типов файлов.
    file_extractor.update(
        {
            ".jpg": image_parser,
            ".png": image_parser,
            ".jpeg": image_parser,
            ".pdf": image_parser,
            ".txt": image_parser,
        }
    )

    return file_extractor

# Получение объекта парсера файлов.
file_extractor = get_file_extractor()

# Функция извлечения терминов из документов.
def extract_terms(documents, term_extract_str, llm_name, model_temperature, api_key):
    # Получение объекта LLM.
    llm = get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

    # Инициализация контекста сервиса.
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=llm),
        prompt_helper=PromptHelper(
            max_input_size=4096, max_chunk_overlap=20, num_output=1024
        ),
        chunk_size_limit=1024,
    )

    # Создание индекса из документов.
    temp_index = GPTListIndex.from_documents(documents, service_context=service_context)

    # Извлечение определений терминов.
    terms_definitions = str(
        temp_index.query(term_extract_str, response_mode="дерево_суммирования")
    )
    # Получение списка строк с определениями.
    terms_definitions = [
        x
        for x in terms_definitions.split("\n")
        if x and "Срок:" in x and "Определение:" in x
    ]
    # Создание словаря с терминами и определениями.
    terms_to_definition = {
        x.split("Определение:")[0]
        .split("Срок:")[-1]
        .strip(): x.split("Определение:")[-1]
        .strip()
        for x in terms_definitions
    }
    return terms_to_definition

# Функция чтения всех файлов в каталоге и подкаталогах и извлечения из них текста.
def read_directory(directory_path):
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root)
with upload_tab:
    st.subheader("Загрузить документы")
    directory_path = st.text_input("Путь к папке", value="./docs")
    if st.button("Прочитать папку"):
        st.spinner("Чтение папки и извлечение текста...")
        documents = read_directory(directory_path)
        st.success(f"Прочитано {len(documents)} документов.")
        if "llama_index" not in st.session_state:
            st.session_state["llama_index"] = initialize_index(
                llm_name, model_temperature, api_key
            )
        terms_extract_str = REFINE_TEMPLATE.format(
            terms=", ".join(st.session_state["all_terms"])
        )
        terms_to_definition = extract_terms(
            documents, terms_extract_str, llm_name, model_temperature, api_key
        )
        insert_terms(terms_to_definition)
        st.success("Успешно извлечены термины и добавлены в индекс.")

with query_tab:
    st.subheader("Запрос")
    query = st.text_input("Введите запрос")
    if st.button("Поиск"):
        if "llama_index" not in st.session_state:
            st.warning("Пожалуйста, загрузите документы и извлеките термины перед поиском.")
            return
        service_context = ServiceContext.from_defaults(
            llm_predictor=LLMPredictor(
                llm=get_llm(llm_name, model_temperature, api_key)
            )
        )
        index = st.session_state["llama_index"]
        results = index.query(query, service_context=service_context)
        for result in results:
            st.write(result.text)
