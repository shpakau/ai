import os
import streamlit as st

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


@st.cache_resource
def get_file_extractor():
    image_parser = ImageParser(keep_image=True, parse_text=True)
    file_extractor = DEFAULT_FILE_EXTRACTOR
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


file_extractor = get_file_extractor()


def extract_terms(documents, term_extract_str, llm_name, model_temperature, api_key):
    llm = get_llm(llm_name, model_temperature, api_key, max_tokens=1024)

    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=llm),
        prompt_helper=PromptHelper(
            max_input_size=4096, max_chunk_overlap=20, num_output=1024
        ),
        chunk_size_limit=1024,
    )

    temp_index = GPTListIndex.from_documents(documents, service_context=service_context)
    terms_definitions = str(
        temp_index.query(term_extract_str, response_mode="дерево_суммирования")
    )
    terms_definitions = [
        x
        for x in terms_definitions.split("\n")
        if x and "Срок:" in x and "Определение:" in x
    ]
    # разобрать текст на дикту
    terms_to_definition = {
        x.split("Определение:")[0]
        .split("Срок:")[-1]
        .strip(): x.split("Определение:")[-1]
        .strip()
        for x in terms_definitions
    }
    return terms_to_definition
def read_directory(directory_path):
    """
     Прочитать все файлы в каталоге и подкаталогах и извлечь из них текст.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        List[Document]: Список документов, где каждый документ соответствует файлу в каталоге..
    """
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in file_extractor:
                with open(file_path, "rb") as f:
                    content = f.read()
                    extracted = file_extractor[file_extension](content)
                    text = extracted.get("text", "")
                    if text.strip():
                        # create the directory "text/postoyanno" if it doesn't exist
                        os.makedirs("text/postoyanno", exist_ok=True)
                        # create the file name by appending ".txt" to the original file name
                        output_file_name = os.path.join("text/postoyanno", file.split(".")[0] + ".txt")
                        # write the extracted text to the output file
                        with open(output_file_name, "w", encoding="utf-8") as output_file:
                            output_file.write(text)
                        documents.append(Document(text))
    return documents



def insert_terms(terms_to_definition):
    for term, definition in terms_to_definition.items():
        doc = Document(f"Срок: {term}\nОпределение: {definition}")
        st.session_state["llama_index"].insert(doc)


@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Создайте объект GPTSQLStructStoreIndex."""
    llm = get_llm(llm_name, model_temperature, api_key)

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm))

    index = GPTSimpleVectorIndex.load_from_disk(
        "./index.json", service_context=service_context
    )

    return index


st.title("Экстрактор терминов индекса Llama 🦙")
st.markdown(
    (
        "Эта программа позволяет загружать ваши собственные документы (либо скриншот/изображение, либо реальный текст) и извлекать термины и определения, создавая базу знаний."
    )
)

setup_tab, terms_tab, upload_tab, query_tab = st.tabs(
    ["Настройка", "Все условия", "Условия загрузки/извлечения", "Условия запроса"]
)

with setup_tab:
    st.subheader("LLM настройка")
    api_key = st.text_input("Введите свой ключ API OpenAI здесь", type="password")
    llm_name = st.selectbox(
        "Какие LLM?", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
    )
    model_temperature = st.slider(
        "Температура LLM", min_value=0.0, max_value=1.0, step=0.1
    )
    term_extract_str = st.text_area(
        "Запрос для извлечения терминов и определений.", value=DEFAULT_TERM_STR
    )
    directory_path = st.text_input("Введите путь к папке для чтения файлов")

    if st.button("Инициализация индекса и сброс условий", key="init_index"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = DEFAULT_TERMS

        if directory_path:
            documents = read_directory(directory_path)
            st.session_state["llama_index"].insert(*documents)



with terms_tab:
    st.subheader("Текущие извлеченные термины и определения")
    st.json(st.session_state["all_terms"])


with upload_tab:
    st.subheader("Определения извлечений и запросов")
    if st.button("Инициализация индекса и сброс условий", key="init_index_1"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = DEFAULT_TERMS

    if "llama_index" in st.session_state:
        st.markdown(
            "Либо загрузите изображение/скриншот документа, либо введите текст вручную."
        )
        uploaded_file = st.file_uploader(
            "Загрузить изображение/скриншот документа:", type=["png", "jpg", "jpeg"]
        )
        document_text = st.text_area("Или введите необработанный текст")
        if st.button("Термины и определения экстракта") and (
            uploaded_file or document_text
        ):
            st.session_state["условия"] = {}
            terms_docs = {}
            with st.spinner("Извлечение (изображения могут быть медленными)..."):
                if document_text:
                    terms_docs.update(
                        extract_terms(
                            [Document(document_text)],
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
                if uploaded_file:
                    Image.open(uploaded_file).convert("RGB").save("temp.png")
                    img_reader = SimpleDirectoryReader(
                        input_files=["temp.png"], file_extractor=file_extractor
                    )
                    img_docs = img_reader.load_data()
                    os.remove("temp.png")
                    terms_docs.update(
                        extract_terms(
                            img_docs,
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
            st.session_state["условия"].update(terms_docs)

    if "условия" in st.session_state and st.session_state["условия"]:
        st.markdown("Extracted terms")
        st.json(st.session_state["условия"])

        if st.button("Insert terms?"):
            with st.spinner("Inserting terms"):
                insert_terms(st.session_state["условия"])
            st.session_state["all_terms"].update(st.session_state["условия"])
            st.session_state["условия"] = {}
            st.experimental_rerun()

with query_tab:
    st.subheader("Запрос терминов/определений!")
    st.markdown(
        (
            "LLM попытается ответить на ваш запрос и дополнит свои ответы, используя введенные вами термины/определения. "
            "Если термина нет в индексе, он ответит, используя свои внутренние знания"
        )
    )
    if st.button("Инициализация индекса и сброс условий", key="init_index_2"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = DEFAULT_TERMS

    if "llama_index" in st.session_state:
        query_text = st.text_input("Спросите о термине или определении:")
        if query_text:
            with st.spinner("Генерация ответа..."):
                response = st.session_state["llama_index"].query(
                    query_text, similarity_top_k=5, response_mode="compact",
                    text_qa_template=TEXT_QA_TEMPLATE, refine_template=REFINE_TEMPLATE
                )
            st.markdown(str(response))
