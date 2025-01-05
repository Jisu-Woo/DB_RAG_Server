# DB_RAG_Server

### Chroma DB update - POST /vectordb/update/{userID}
-> 해당 userID의 vector DB collection을 MySQL과 동기화

### Query retrieval - POST /query
-> 입력받은 query에 대해 특정 유저의 collection 내에서 유사도 높은 내용을 retrieval하고, 해당 내용 내에서 gpt-4o 모델을 사용하여 자연스러운 답변을 생성하여 리턴

- Request Header
headers = {
    "Content-Type": "application/json"
}

- Request Body example
{
    "query": "데이터마이닝에 대해서 알려줘.",
    "user_id": "jisu"
}
