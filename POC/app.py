import os

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from chain_utils import get_chain_by_file
from datetime import datetime
import logging

USER_AVATAR = "👤"
BOT_AVATAR = "🦝"
import time
img_path = "./exports/charts/temp_chart.png"
file_is_not_generated = "file is not generated"

#---
logger = logging.getLogger("LLM logger")
logger.setLevel(logging.DEBUG)

# remove all default handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

# create console handler and set level to debug
console_handle = logging.StreamHandler()
console_handle.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(name)-20s - %(levelname)-8s - %(message)s")
console_handle.setFormatter(formatter)

# now add new handler to logger
logger.addHandler(console_handle)
# ---


def set_chat_display_module():
    st.header("GENAI POC - Excel QnA")
    chain = st.session_state["chain"]
    if chain is None:
        st.info("Upload your Excel filed first.")
        return

    if "message" not in st.session_state.keys():
        st.session_state.message = [
            {"role": "assistant", "content": "Hello, I am a raccoon. Ask me a question!"}
        ]

    if prompt := st.chat_input("Your question"):
        st.session_state.message.append({"role": "user", "content": prompt})

    for message in st.session_state.message:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
            if type(message["content"]) == str and 'exports/charts' in message["content"]:
                st.image(message["content"])

    if st.session_state.message[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="🦝"):
            with st.spinner("Responding..."):
                time_start = datetime.now()
                response = chain.chat(prompt)
                time_finish = datetime.now()
                st.write(response)
                if type(response) == str and 'exports/charts' in response:
                    time_to_wait = 4
                    time_counter = 0
                    while not os.path.exists(response):
                        time.sleep(1)
                        time_counter += 1
                        logger.debug("waiting for file...")
                        if time_counter > time_to_wait:
                            logger.debug(file_is_not_generated)
                            response = file_is_not_generated
                            break
                    if file_is_not_generated not in response:
                        st.image(response)
                    else:
                        st.write(response)
                logger.info("took (sec):" + str(int((time_finish - time_start).total_seconds())))
                message = {"role": "assistant", "content": response}
                st.session_state.message.append(message)


def set_sidebar_settings_module():
    with st.sidebar:
        st.header("Settings")
        st.write("Define PANDASAI API key, if not defined \n LocalLLM - 'codelama' will be used")
        openai_api_key = st.text_input(label="- PANDASAI API key", type="password")

        if st.button(label="Save settings"):
            os.environ["PANDASAI_API_KEY"] = openai_api_key
            st.success("✅ Settings saved!")


def set_sidebar_upload_xlsx_module():
    with st.sidebar:
        st.header("Feed your documents here")
        st.session_state["xlsx_docs"].clear()
        st.session_state["xlsx_docs"].extend(
            st.file_uploader(
                label="Upload your documents here",
                type=["xlsx", "csv"],
                accept_multiple_files=True,
            )
        )


def get_chain_when_pressing_process_button():
    with st.sidebar:
        if len(st.session_state["xlsx_docs"]) == 0:
            st.button(label="Start Chunking", disabled=True)
        else:
            if st.button(label="Start Chunking", disabled=False, key="process_button"):
                with st.spinner("Processing"):
                    st.session_state["chain"] = get_chain_by_file()
                if st.session_state["chain"] is None:
                    st.error("❌ Processing failed.")
                    return
                st.success("✅ File successfully processed!")
                st.session_state["xlsxs_processed"] = True


def main():
    # set default values
    if "xlsx_docs" not in st.session_state:
        st.session_state["xlsx_docs"] = []
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "current_image" not in st.session_state:
        st.session_state["current_image"] = None
    if "xlsxs_processed" not in st.session_state:
        st.session_state["xlsx_processed"] = False
    if "langchain_messages" not in st.session_state:
        st.session_state["langchain_messages"] = StreamlitChatMessageHistory()

    st.set_page_config(page_title="Excel QnA", page_icon=":notebook:")
    set_sidebar_settings_module()
    set_sidebar_upload_xlsx_module()
    get_chain_when_pressing_process_button()
    set_chat_display_module()


if __name__ == "__main__":
    main()
