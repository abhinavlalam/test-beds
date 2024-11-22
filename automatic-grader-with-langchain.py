# cannot run this as there is no free api from OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

from sk import my_sk #importing secret key from another python file

