import json
import math
import pandas as pd
from ibm_watson.discovery_v2 import DiscoveryV2, QueryLargePassages
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os, re
from dotenv import load_dotenv

class WatsonQA:

    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path)

        self.WD_API_KEY = os.getenv('WD_API_KEY')
        self.WD_PROJECT_ID = os.getenv('WD_PROJECT_ID')
        self.WD_PROJECT_ID_2 = os.getenv('WD_PROJECT_ID_2')
        self.WD_URL = os.getenv('WD_URL')

        self.WX_API_KEY = os.getenv('WX_API_KEY')
        self.WX_PROJECT_ID = os.getenv('WX_PROJECT_ID')
        self.WX_URL = os.getenv('WX_URL')

        # self.WD_API_KEY = os.environ['WD_API_KEY']
        # self.WD_PROJECT_ID = os.environ['WD_PROJECT_ID']
        # self.WD_PROJECT_ID_2 = os.environ['WD_PROJECT_ID_2']
        # self.WD_URL = os.environ['WD_URL']

        # self.WX_API_KEY = os.environ['WX_API_KEY']
        # self.WX_PROJECT_ID = os.environ['WX_PROJECT_ID']
        # self.WX_URL = os.environ['WX_URL']

        # Initialize Watson Discovery
        self.authenticator_wd = IAMAuthenticator(self.WD_API_KEY)
        self.discovery = DiscoveryV2(
            version='2019-04-30',
            authenticator=self.authenticator_wd
        )
        self.discovery.set_service_url(self.WD_URL)

        # Initialize Watson XAI
        self.api_key_wx = self.WX_API_KEY
        self.ibm_cloud_url_wx = self.WX_URL
        self.project_id_wx = self.WX_PROJECT_ID
        self.creds_wx = {
            "url": self.ibm_cloud_url_wx,
            "apikey": self.api_key_wx
        }

    def send_to_watsondiscovery(self, user_question, PROJECT_ID, text_list):
        authenticator = IAMAuthenticator(self.WD_API_KEY)
        discovery = DiscoveryV2(
            version='2019-04-30',
            authenticator=authenticator
        )
        discovery.set_service_url(self.WD_URL)

        collections = discovery.list_collections(project_id=PROJECT_ID).get_result()
        collection_list = list(pd.DataFrame(collections['collections'])['collection_id'])

        total_pages=10
        passages = QueryLargePassages(per_document=True, find_answers=True, max_per_document=total_pages)

        query_result = discovery.query(
            project_id=PROJECT_ID,
            collection_ids=collection_list,
            natural_language_query=user_question,
            passages=passages).get_result()
        
        # Set wording or passage
        # text_list=False

        if text_list == True:
            start_offset = [math.floor(query_result['results'][i]['document_passages'][0]['start_offset'] / 1000) * 1000 for i in
                            range(len(query_result['results']))]
            end_offset = [math.ceil(query_result['results'][i]['document_passages'][0]['end_offset'] / 1000) * 1068 for i in
                            range(len(query_result['results']))]
            # passages_list = [query_result['results'][i]['document_passages'][0]['passage_text'] for i in range(len(query_result['results']))]
            text_list = [query_result['results'][i]['text'][0] for i in range(len(query_result['results']))]

            passage_index = 0  # Initialize passage index
            len_text = len(text_list[passage_index])
            context_text = text_list[passage_index][
                           start_offset[passage_index]:min(end_offset[passage_index], len_text)]

        else:
            ### Select best highest confidence passage
            # passage_texts = []  # Initialize an empty list to store passage texts
            # max_confidence_index = None  # Initialize a variable for highest confidence

            # for i in range(len(query_result['results'][0]['document_passages'])):
            #     confidence = query_result['results'][0]['document_passages'][i]['answers'][0]['confidence']

            #     # Check if the current confidence is higher than the previous maximum confidence
            #     if max_confidence_index is None or confidence > query_result['results'][0]['document_passages'][max_confidence_index]['answers'][0]['confidence']:
            #         max_confidence_index = i  # Update the index with the highest confidence

            # # Append the passage text with the highest confidence to the list
            # max_confidence = query_result['results'][0]['document_passages'][max_confidence_index]['answers'][0]['confidence']
            # print(f"Score WD: {max_confidence}")
            # passage_texts.append(query_result['results'][0]['document_passages'][max_confidence_index]['passage_text'])
            # combined_text = ' '.join(passage_texts)
            # context_text = re.sub(r'<\/?em>', '', combined_text)

            ### Select best n passages
            passage_texts = []  # Initialize an empty list to store passage texts
            confidence_scores = []  # Initialize a list to store confidence scores

            sorted_passages = sorted(query_result['results'][0]['document_passages'], key=lambda x: x['answers'][0]['confidence'], reverse=True)

            # Take the top 3 passages with the highest confidence scores
            for i in range(min(3, len(sorted_passages))):
                passage_text = sorted_passages[i]['passage_text']
                confidence = sorted_passages[i]['answers'][0]['confidence']
                
                print(f"Score WD {i + 1}: {confidence}")
                
                passage_texts.append(passage_text)
                confidence_scores.append(confidence)

            combined_text = ' '.join(passage_texts)
            context_text = re.sub(r'<\/?em>', '', combined_text)
            
        context_text = re.sub(r'"(\n\n)', '', context_text)
        print(f"context_text:\n{context_text}\n")
        return context_text

    def send_to_watsonxai(self, prompts, model_name='meta-llama/llama-2-13b-chat', decoding_method="greedy",
                          max_new_tokens=4096, min_new_tokens=1, temperature=0, repetition_penalty=1.0,
                          stop_sequences=["\n\n"]):
        assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"

        model_params = {
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.RANDOM_SEED: 42,
            GenParams.TEMPERATURE: temperature,
            GenParams.REPETITION_PENALTY: repetition_penalty,
            GenParams.STOP_SEQUENCES: stop_sequences
        }

        api_key = self.WX_API_KEY
        ibm_cloud_url = self.WX_URL
        project_id = self.WX_PROJECT_ID

        creds = {
            "url": ibm_cloud_url,
            "apikey": api_key
        }

        model = Model(
            model_id=model_name,
            params=model_params,
            credentials=creds,
            project_id=project_id)

        for prompt in prompts:
            output = model.generate_text(prompt)

        return output
    
    ### Show promo recomendation based on BNI products offer
    def watsonxai_promo(self, user_question, context_previous):
        
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=True)
        # json_format = {"nama promo":" ", "tipe_kartu":" ", "persyaratan":" "}
        json_format = {"alt": "nama promo", "url": "jpg url jika ada", "title": "nama promo", "description": "deskripsi dan persyaratan promo"}

        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        Konteks_wd: {context_text}
        Konteks_previous: {context_previous}
        Pertanyaan: {user_question}
        Berikan 3 rekomendasi promo BNI berdasarkan kriteria dari Konteks_previous yang terkait dengan Konteks_wd berupa json format: {json_format}. Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        # output_stage["output"] = re.sub("\n", "", output_stage["output"])

        return output_stage
    
    ### Show how many cards available based on BNI promo offer
    def watsonxai_product(self, user_question, context_previous):

        PROJECT_ID = self.WD_PROJECT_ID_2
        context_text = self.send_to_watsondiscovery(user_question, PROJECT_ID, text_list=False)
        # json_format = {"nama promo":" ", "tipe_kartu":" ", "persyaratan":" "}
        json_format = {"alt": "produk kartu bni", "url": "jpg url jika ada", "title": "produk kartu bni", "description": "deskripsi dan persyaratan produk kartu bni"}

        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        Konteks_wd: {context_text}
        Konteks_previous: {context_previous}
        Pertanyaan: {user_question}
        Carilah informasi produk BNI yang sesuai dengan Pertanyaan. Berikan informasi produk BNI berdasarkan kriteria dari Konteks_previous yang terkait dengan Konteks_wd berupa json format: {json_format}. Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        # output_stage["output"] = re.sub("\n", "", output_stage["output"])

        return output_stage
    
    ### Show information about the products
    def watsonxai_product_information(self, user_question):

        PROJECT_ID = self.WD_PROJECT_ID_2
        context_text = self.send_to_watsondiscovery(user_question, PROJECT_ID, text_list=False)

        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        Konteks_wd: {context_text}
        Pertanyaan: {user_question}
        Berikan informasi produk BNI berdasarkan kriteria dari Konteks_wd. 
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        # output_stage["output"] = re.sub("\n", "", output_stage["output"])

        return output_stage
    
# Example Usage
# watson_qa_instance = WatsonQA()
# result = watson_qa_instance.watsonxai(user_question, system_prompt)
# print(result)