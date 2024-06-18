
import os
import tempfile
from typing import List
import pandas as pd
from langchain_community.llms.ollama import Ollama
from pandas import DataFrame
from pandasai import SmartDataframe, skill
import streamlit as st
from pandasai.llm.local_llm import LocalLLM


def get_data_from_xlsx() -> List[DataFrame]:
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in st.session_state["xlsx_docs"]:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        function = pd.read_csv if temp_filepath.endswith(".csv") else pd.read_excel
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        data = function(temp_filepath)
        docs.extend(data)
        print(data.shape)
        print(data.columns.tolist())
        # TODO: plan to work with several not with one
    return data


def get_llm() -> LocalLLM:
    return Ollama(model='llama3:8b-instruct-q5_K_M', temperature=0)
    # return Ollama(model='llama3:8b-instruct-q5_K_M')
    # return Ollama(model='phi3')
    # return Ollama(model='llama3:8b')
    # return LlamaCpp(model_path="./models/codellama-7b-instruct.Q4_K_M.gguf", max_tokens=2000, n_ctx=2048)
    # return LocalLLM(api_base="http://localhost:11434/v1", model="codellama")


@skill
def plot_sales_amount(names: list[str], salaries: list[int]):
    """
    Displays the bar chart having name on x-axis and salaries on y-axis using streamlit
    Args:
        names (list[str]): Employees' names
        salaries (list[int]): Salaries
    """
    import matplotlib.pyplot as plt
    print("___skill is used___")
    plt.bar(names, salaries)
    plt.xlabel("Sales Amount")
    plt.ylabel("Category")
    plt.title("Sales amount by category")
    plt.xticks(rotation=45)
    plt.savefig("temp_chart.png")
    fig = plt.gcf()
    # st.pyplot(fig)
    st.write(fig)


def get_chain_by_file():
    docs = get_data_from_xlsx()
    print(f"Dataset = \n{docs}")
    if os.environ['PANDASAI_API_KEY']:
        print("Going with pandas API LLM")
        df = SmartDataframe(docs, {"enable_cache": False, "save_charts": True,})
        # df = Agent([docs], memory_size=10, config={"verbose": True, "temperature": 0, "enable_cache": False})
    else:
        df = SmartDataframe(docs, config={"llm": get_llm(), "verbose": True,})
    df.add_skills(plot_sales_amount)
    return df

