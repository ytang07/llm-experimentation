import os

from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from pymilvus import connections, utility
from dotenv import load_dotenv

load_dotenv()

zilliz_uri = os.getenv("ZILLIZ_CLUSTER_01_URI")
zilliz_token = os.getenv("ZILLIZ_CLUSTER_01_TOKEN")

headers_to_split_on = [
    ("##", "Section"),
]
path='./notion_docs'

default_chunk_size = 64
default_chunk_overlap = 8

def test_langchain_chunking(docs_path, splitters, chunk_size, chunk_overlap, drop_collection=True):

    path=docs_path
    loader = NotionDirectoryLoader(path)
    docs = loader.load()
    md_file=docs[0].page_content

    # Let's create groups based on the section headers in our page
    headers_to_split_on = splitters
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(md_file)

    # Define our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(md_header_splits)

    test_collection_name = f"EngineeringNotionDoc_{chunk_size}_{chunk_overlap}"

    vectordb = Milvus.from_documents(documents=all_splits,
                                    embedding=OpenAIEmbeddings(),
                                    connection_args={"uri": zilliz_uri, 
                                                    "token": zilliz_token},
                                    collection_name=test_collection_name)

    metadata_fields_info = [
        AttributeInfo(
            name="Section",
            description="Part of the document that the text comes from",
            type="string or list[string]"
        ),
    ]
    document_content_description = "Major sections of the document"

    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(llm, vectordb, document_content_description, metadata_fields_info, verbose=True)

    res = retriever.get_relevant_documents("What makes a distinguished engineer?")
    print(f"""Responses from chunking strategy:
          {chunk_size}, {chunk_overlap}""")
    for doc in res:
        print(doc)

    # this is just for rough cleanup, we can improve this
    # lots of user considerations to understand for real experimentation use cases though
    if drop_collection:
        connections.connect(uri=zilliz_uri, token=zilliz_token)
        utility.drop_collection(test_collection_name)   

chunking_tests = [(32, 4), (64, 8), (128, 16), (256, 32), (512, 64)]
for test in chunking_tests:
    test_langchain_chunking(path, headers_to_split_on, test[0], test[1])