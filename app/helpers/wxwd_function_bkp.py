import json
import math
import pandas as pd
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

#===========================Load env=====================================
import os
# from dotenv import load_dotenv
# dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
# load_dotenv(dotenv_path)

# WD_API_KEY = os.getenv('WD_API_KEY')
# WD_PROJECT_ID = os.getenv('WD_PROJECT_ID')
# WD_URL = os.getenv('WD_URL')

# WX_API_KEY = os.getenv('WX_API_KEY')
# WX_PROJECT_ID = os.getenv('WX_PROJECT_ID')
# WX_URL = os.getenv('WX_URL')

WD_API_KEY = os.environ['WD_API_KEY']
WD_PROJECT_ID = os.environ['WD_PROJECT_ID']
WD_URL = os.environ['WD_URL']

WX_API_KEY = os.environ['WX_API_KEY']
WX_PROJECT_ID = os.environ['WX_PROJECT_ID']
WX_URL = os.environ['WX_URL']

#===========================main function=====================================
def main(params):
    # user_question = params['user_question']
    user_question = params
    authenticator = IAMAuthenticator(WD_API_KEY)
    discovery = DiscoveryV2(
        version='2019-04-30',
        authenticator=authenticator
    )
    discovery.set_service_url(WD_URL)

    #discovery.set_disable_ssl_verification(True)

    PROJECT_ID = WD_PROJECT_ID
    ## List Collections ##
    collections = discovery.list_collections(project_id=PROJECT_ID).get_result()
    collection_list = list(pd.DataFrame(collections['collections'])['collection_id'])

    query_result = discovery.query(
        project_id=PROJECT_ID,
        collection_ids=collection_list,
        natural_language_query=user_question).get_result()
    
    start_offset = [math.floor(query_result['results'][i]['document_passages'][0]['start_offset']/1000)*1000 for i in range(len(query_result['results']))]
    end_offset = [math.ceil(query_result['results'][i]['document_passages'][0]['end_offset']/1000)*1000 for i in range(len(query_result['results']))]
    passages_list = [query_result['results'][i]['document_passages'][0]['passage_text'] for i in range(len(query_result['results']))]
    text_list = [query_result['results'][i]['text'][0] for i in range(len(query_result['results']))]

    # First Prompt
    format_stage1 = '{"passage_number":, "reason":}'
    prompt_stage1 = f"""context: {passages_list}

    question: {user_question}

    The context provided in an array of passages.
    Please select the number of the passage that possibly answers the question. 
    Output the number and the reason in a json format.
    The output example is: {format_stage1}
    Avoid adding any character such as space, tab, or newline.
    Stop generating any additional information beyond this point.

    output:"""

    passage_index = 0 #Initialize passage index
    output_stage1 = send_to_watsonxai(prompts=[prompt_stage1])
    try:
        passage_index = max(int(json.loads(output_stage1.strip())['passage_number'])-1,0)
        passage_index = min(passage_index,len(text_list)-1)
        print(passage_index)
        print(len(text_list))
    except: 
        print(output_stage1)

    len_text = len(text_list[passage_index])
    context_text = text_list[passage_index][start_offset[passage_index]:min(end_offset[passage_index],len_text)]

    # Second prompt
    format_stage2 = '{"output":}'
    prompt_stage2 = f"""passage: {context_text}

    question: {user_question}

    Answer the question only using the passage above as a context. 
    Compile the answer in an engaging way. Summarize the answer in a concise way.

    answer:"""

    output_stage2 = send_to_watsonxai(prompts=[prompt_stage2], stop_sequences=[])

    return {"output":str(output_stage2.strip()).replace('\n\n', ' ').replace('*', '<li>')}


#==============================HELPER FUNCTION======================================

def send_to_watsonxai(prompts,
                    model_name='meta-llama/llama-2-70b-chat',
                    decoding_method="greedy",
                    max_new_tokens=1000,
                    min_new_tokens=1,
                    temperature=0,
                    repetition_penalty=1.0,
                    stop_sequences=["\n\n"]
                    
                    ):
    '''
   helper function for sending prompts and params to Watsonx.ai
    
    Args:  
        prompts:list list of text prompts
        decoding:str Watsonx.ai parameter "sample" or "greedy"
        max_new_tok:int Watsonx.ai parameter for max new tokens/response returned
        temperature:float Watsonx.ai parameter for temperature (range 0>2)
        repetition_penalty:float Watsonx.ai parameter for repetition penalty (range 1.0 to 2.0)

    Returns: None
        prints response
    '''

    assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"

    # Instantiate parameters for text generation
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    api_key =  WX_API_KEY   #IBM Cloud API Key
    ibm_cloud_url = WX_URL
    project_id = WX_PROJECT_ID #Project ID watsox.ai

    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)


    for prompt in prompts:
        output = model.generate_text(prompt)

    return output
