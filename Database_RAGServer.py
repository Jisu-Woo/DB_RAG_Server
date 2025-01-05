from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import mysql.connector
import os
import openai
from chromadb import Client
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pymysql
import chromadb.utils.embedding_functions as embedding_functions
from pydantic import BaseModel
from typing import Dict, Any


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002"
)


# FastAPI 인스턴스 생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)




# Database를 저장할 경로 지정
directory = "path"

client = chromadb.PersistentClient(path=directory)


# MySQL 연결 설정
MYSQL_CONFIG = {
    "host": "",
    "user": "",
    "password": "",
    "database": "",
    "port":,
}


# 요청 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str
    user_id: str


# MySQL에서 데이터 가져오기
def fetch_posts_from_mysql(user_id):
    try:
        # pymysql 연결
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor(
            pymysql.cursors.DictCursor
        )  # Dictionary 형태로 결과 반환
        query = """
            SELECT postID, title, content
            FROM post
            WHERE userID = %s
        """
        cursor.execute(query, (user_id,))
        posts = cursor.fetchall()
        return posts
    except pymysql.MySQLError as e:
        # pymysql 오류 처리
        raise HTTPException(status_code=500, detail=f"MySQL Error: {e}")
    finally:
        if "connection" in locals() and connection.open:
            connection.close()
            print("MySQL 연결 종료.")


def update_chroma_collection(user_id, posts):

    # 컬렉션 이름 정의
    collection_name = f"user_{user_id}_posts"

    try:
        # 기존 컬렉션 가져오기
        existing_collections = [col.name for col in client.list_collections()]

        # 컬렉션이 존재하면 삭제
        if collection_name in existing_collections:
            client.delete_collection(name=collection_name)
            print(f"기존 컬렉션 '{collection_name}' 삭제 완료.")

        # 새 컬렉션 생성
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"컬렉션 '{collection_name}' 생성 완료.")

        # Document 객체 생성
        documents = [
            Document(
                page_content=post["content"],
                metadata={
                    "post_id": post["postID"],
                    "title": post["title"],
                },
            )
            for post in posts
        ]

        # 필요한 리스트로 변환
        document_contents = [doc.page_content for doc in documents]
        document_metadatas = [doc.metadata for doc in documents]
        document_ids = [str(doc.metadata["post_id"]) for doc in documents]

        # 문서 추가
        try:
            collection.add(
                documents=document_contents,
                metadatas=document_metadatas,
                ids=document_ids,
            )

            return {
                "status": "success",
                "message": f"Collection '{collection_name}' updated successfully",
            }

        except Exception as e:
            return {"status": "error", "message": f"문서 추가 중 오류 발생: {str(e)}"}

    except Exception as e:
        print(f"컬렉션 생성 또는 삭제 중 오류 발생: {str(e)}")
        return None



@app.post("/vectordb/update/{user_id}")
def update_user_posts(user_id: str):
    # MySQL에서 포스트 데이터 가져오기
    posts = fetch_posts_from_mysql(user_id)
    if not posts:
        raise HTTPException(
            status_code=404, detail="No posts found for the given user ID."
        )

    # ChromaDB 콜렉션 업데이트
    result = update_chroma_collection(user_id, posts)

    return result


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"message": "Internal server error"})



# 사용자의 컬렉션에서 검색 수행
def retrieve_from_collection(user_id: str, query: str) -> str:
    global openai_ef
    # 컬렉션 이름 설정
    collection_name = f"user_{user_id}_posts"

    try:
        # 기존 컬렉션 가져오기
        collection = client.get_collection(
            name=collection_name, embedding_function=openai_ef
        )
        print("콜렉션 가져오기")
        print(openai_ef)
        print(query)
        query_result = collection.query(
            query_texts=[query],
            n_results=2,
        )
        print(query_result)

        # OpenAI API에 쿼리 연결 ---------------------------------------

        documents = query_result.get("documents", [])
        metadatas = query_result.get("metadatas", [])

        retrieved_docs = " ".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(documents)]
        )

        metadata_info = " ".join(
            [f"Document {i+1} metadata: {meta}" for i, meta in enumerate(metadatas)]
        )

        # 프롬프트 생성
        system_prompt = """You are an AI assistant that helps users understand documents and answer questions based on them. 
                            Caution: Always generate responses solely based on the provided document content. Respond only in Korean."""
        user_prompt = f"""
        아래는 검색된 관련 문서입니다:

        {retrieved_docs}

        이 문서들과 관련된 메타데이터는 다음과 같습니다:
        {metadata_info}

        질문: {query}

        오직 위 문서만을 참고하여 질문에 대한 답을 생성해주세요.
        """

        # OpenAI ChatCompletion 호출
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        response_content = response.choices[0].message.content
        # 결과 출력
        print("답변:", response_content)

        return response_content
        

    except Exception as e:
        return {"status": "error", "message": f"Retrieval 중 오류 발생: {str(e)}"}



@app.post("/query", response_model=Dict[str, Any])
def query_collection(request: QueryRequest):
    try:
        # retrieve_from_collection 호출
        response_content = retrieve_from_collection(request.user_id, request.query)
        return {
            "status": "success",
            "user_id": request.user_id,
            "query": request.query,
            "answer": response_content,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")
