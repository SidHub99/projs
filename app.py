import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


def myfunc(topic,words,role):
    llm=CTransformers(model='meta-llama/Llama-2-7b',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.03})
    template="""
            Write me a blog for topic {topic} with maximum words of {words} specifically for {role}
            """
    prompt=PromptTemplate(input_variables=[topic,words,role],template= template)
    result=llm(prompt.format(role=role,words=words,topic=topic))
    print(result)
    return result


st.set_page_config(page_title="Blog generation",
layout='centered',
initial_sidebar_state='collapsed'
)
st.header("BLOGGING")
topic=st.text_input("Tell Us what you want to generate")
col1,col2=st.columns([5,5])
with col1:
    words=st.text_input("NO of words")
with col2:
    role=st.selectbox('Write a blog for',("Common man","Researcher","IT person"),index=0)

submit=st.button('Generate')

if submit:
    st.write(myfunc(topic,words,role))