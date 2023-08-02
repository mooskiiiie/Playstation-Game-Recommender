import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('PlayStation Game Recommender')
prompt = st.text_input('Let me know what kind of games you like, and I will recommend you some games based on them')

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'You are the ultimate gamer. Recommend 5 playstation games based on {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title'],
    template = 'Tell me why these games are great for a gamer based on the title TITLE: {title}'
)


# LLMS
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'])

if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])
