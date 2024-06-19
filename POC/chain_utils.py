import os
import matplotlib.pyplot as plt
import matplotlib
import tempfile
from typing import List
import pandas as pd
from langchain_community.llms.ollama import Ollama
from pandas import DataFrame
from pandasai import SmartDataframe, Agent
from pandasai.skills import skill
import streamlit as st
from pandasai.llm.local_llm import LocalLLM


def get_data_from_spreadsheet() -> List[DataFrame]:
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in st.session_state["xlsx_docs"]:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        function = pd.read_csv if temp_filepath.endswith(".csv") else pd.read_excel
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        data = function(temp_filepath)
        docs.extend(data)
    return data


def get_llm() -> LocalLLM:
    return Ollama(model='llama3:8b-instruct-q5_K_M')


@skill
def plot_sales_amount(sales_amount: list[int], categories: list[str]):
    """
    Displays the bar chart having name on x-axis and salaries on y-axis using streamlit
    Args:
        sales_amount (list[str]): Sales Amount
        categories (list[int]): Categories
    """
    print("... skills is used ... ")
    plt.bar(sales_amount, categories)
    plt.xlabel("Sales")
    plt.ylabel("Category")
    plt.title("Sales amount")
    plt.xticks(rotation=45)
    plt.savefig("temp_chart.png")
    fig = plt.gcf()
    st.pyplot(fig)
    st.session_state["current_image"] = fig


def get_chain_by_file():
    docs = get_data_from_spreadsheet()
    initial_conf = {"verbose": True, "temperature": 0, "enable_cache": False, "open_charts": False,
                    "save_charts": True}
    local_llm = {"llm": get_llm()}
    if os.environ['PANDASAI_API_KEY']:
        agent = Agent([docs], memory_size=10,
                      config=initial_conf)

    else:
        agent = SmartDataframe(docs, config=initial_conf.update(local_llm))
    agent.add_skills(plot_sales_amount)
    return agent
