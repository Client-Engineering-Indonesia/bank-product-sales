import warnings
warnings.filterwarnings('ignore')

import io
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
import pandas as pd

from app.helpers.helper import *
from app.helpers.wxwd_function import *

app = FastAPI(
    title='Sample-app FastAPI and Docker',
    version = '1.0.0',
)


@app.get("/")
async def root():
    return {"message": "Hello World with BNI"}

@app.get("/ping")
async def ping():
    return "Hello, I am alive..."

# @app.post("/process_dict")
# async def process_dict(input_dict: dict):
#     # You can perform any processing on the input dictionary here
#     # For example, let's just return the received dictionary as is
#     return input_dict

@app.post("/bni_product_reco")
async def get_recommendation(request: Request):

    try:
        user_input = await request.json()
        #question = user_input['user_question']
        context = user_input['cust_profile']
        watson_qa_instance = WatsonQA()
        answer = await watson_qa_instance.watsonxai_product_reco(context)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/bni_promo")
async def get_promo(request: Request):
    #user_question, context_previous
    try:
        user_input = await request.json()
        product_name = user_input['product_name']
        context = user_input['cust_profile']
        watson_qa_instance = WatsonQA()
        answer = await watson_qa_instance.watsonxai_product_promo(product_name, context)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bni_product_info")
async def get_product_info(request: Request):

    try:
        user_input = await request.json()
        question = user_input['user_question']
        watson_qa_instance = WatsonQA()
        answer = await watson_qa_instance.watsonxai_product_information(question)
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Custom title",
#         version="3.0.2",
#         summary="This is a very custom OpenAPI schema",
#         description="Here's a longer description of the custom **OpenAPI** schema",
#         routes=app.routes,
#     )
#     openapi_schema["info"]["x-logo"] = {
#         "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
#     }
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="3.0.2",
        description="Here's a longer description of the custom **OpenAPI** schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    openapi_schema["servers"] = [{"url": "http://localhost:8000"}]  # Add your server URL
    app.openapi_schema = openapi_schema
    return app.openapi_schema



app.openapi = custom_openapi