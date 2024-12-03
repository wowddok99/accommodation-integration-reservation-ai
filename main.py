import tensorflow as tf
import numpy as np
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,Response
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


import uvicorn

model = tf.keras.models.load_model('./train_model.h5', compile=False)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
result = [
    '모츠나베',
    '부대찌개',
    '닭볶음탕',
    '바베큐',
    '대파닭꼬치',
    '두루치기',
    '어묵탕',
    '홍합탕',
    '매운탕',
    '감바스',
    '조개/새우구이',
    '골뱅이무침&소면',
    '밀푀유 나베',
    '샤브샤브&칼국수',
    '파스타',
    '소세지 야채볶음',
    '타코'
    ]
@app.get("/predict")
def predict(
    material: str = Query(...),  # 필수 매개변수
    soup: str = Query(...),      # 필수 매개변수
    like_taste: str = Query(...),  # 기본값 설정
    cook_level: str = Query(...)      # 기본값 설정
):
    try:
        arr = [0,0,0,0]
        if material=="meat":
            arr[0] = 1
        if material=="seafood":
            arr[0] = 2
        if material=="idontknow":
            arr[0] = 3

        if soup=="yes":
            arr[1] = 1
        if soup=="no":
            arr[1] = 2

        if like_taste=="spicy":
            arr[2] = 3
        if like_taste=="plain":
            arr[2] = 2
        if like_taste=="greasy":
            arr[2] = 1

        if cook_level=="good":
            arr[3] = 3
        if cook_level=="soso":
            arr[3] = 2
        if cook_level=="bad":
            arr[3] = 1




        
        value_list = np.array(arr, dtype=np.float32).reshape(1, -1)
        predict_result = model.predict(value_list)
        predict_class = np.argmax(predict_result, axis=1)[0]
        return JSONResponse(content={"recommended_menu":result[int(predict_class)],"recommended_menu_number":int(predict_class)},status_code=200)
    except ValueError:
        return Response(content="Invalid Value",status_code=400)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8060,reload=True)