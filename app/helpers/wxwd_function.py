import json
import math
import pandas as pd
from ibm_watson.discovery_v2 import DiscoveryV2, QueryLargePassages
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os, re, ast
# from dotenv import load_dotenv

class WatsonQA:

    def __init__(self):
        # dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        # load_dotenv(dotenv_path)

        # self.WD_API_KEY = os.getenv('WD_API_KEY')
        # self.WD_PROJECT_ID = os.getenv('WD_PROJECT_ID')
        # self.WD_PROJECT_ID_2 = os.getenv('WD_PROJECT_ID_2')
        # self.WD_PROJECT_ID_3 = os.getenv('WD_PROJECT_ID_3')
        # self.WD_URL = os.getenv('WD_URL')

        # self.WX_API_KEY = os.getenv('WX_API_KEY')
        # self.WX_PROJECT_ID = os.getenv('WX_PROJECT_ID')
        # self.WX_URL = os.getenv('WX_URL')

        self.WD_API_KEY = os.environ['WD_API_KEY']
        self.WD_PROJECT_ID = os.environ['WD_PROJECT_ID']
        self.WD_PROJECT_ID_2 = os.environ['WD_PROJECT_ID_2']
        self.WD_PROJECT_ID_3 = os.environ['WD_PROJECT_ID_3']
        self.WD_URL = os.environ['WD_URL']

        self.WX_API_KEY = os.environ['WX_API_KEY']
        self.WX_PROJECT_ID = os.environ['WX_PROJECT_ID']
        self.WX_URL = os.environ['WX_URL']

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
            end_offset = [math.ceil(query_result['results'][i]['document_passages'][0]['end_offset'] / 1000) * 1000 for i in
                            range(len(query_result['results']))]

            # text_list = [query_result['results'][i]['text'][0] for i in range(len(query_result['results']))]

            # passage_index = 0  # Initialize passage index
            # len_text = len(text_list[passage_index])
            # context_text = text_list[passage_index][
            #                start_offset[passage_index]:min(end_offset[passage_index], len_text)]

            max_allowed_length = 3000

            adjusted_start_offsets = []
            adjusted_end_offsets = []

            for start_offset, end_offset in zip(start_offset, end_offset):
                length = end_offset - start_offset
                if length > max_allowed_length:
                    # Adjust the offsets to ensure the length is no more than max_allowed_length
                    adjusted_end_offset = start_offset + max_allowed_length
                    adjusted_start_offsets.append(start_offset)
                    adjusted_end_offsets.append(adjusted_end_offset)
                else:
                    adjusted_start_offsets.append(start_offset)
                    adjusted_end_offsets.append(end_offset)

            text_list = [query_result['results'][i]['text'][0] for i in range(len(query_result['results']))]

            passage_index = 0  # Initialize passage index
            len_text = len(text_list[passage_index])
            context_text = text_list[passage_index][
                            adjusted_start_offsets[passage_index]:min(adjusted_end_offsets[passage_index], len_text)]

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
    


    async def watsonxai_product_reco(self, context_previous):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=False)
        json_format = {
            "alt": "nama produk BNI",
            "url": "jpg url jika ada",
            "title": "nama produk BNI, bisa berupa nama tabungan, kartu debit, atau kartu kredit",
            "description": "deskripsi dan persyaratan dari produk"
        }
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        promo: {context_text}
        client: {context_previous}
        Temukan 3 promo yang sesuai dengan info profile client. Ekstrak 3 nama produk bank BNI seperti nama tabungan, nama kartu kredit, nama kartu debit yang ada pada promo. Cari informasi dari produk tersebut. Outputkan hasil dalam format JSON seperti {json_format}."
        Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Output:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        output_stage["output"] = ast.literal_eval(output_stage['output'])
        return output_stage
    

    async def watsonxai_product_reco_crsl(self, context_previous):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=False)
        json_format = {
            "alt": "nama produk BNI",
            "url": "jpg url jika ada",
            "title": "nama produk BNI, bisa berupa nama tabungan, kartu debit, atau kartu kredit",
            "description": "deskripsi dan persyaratan dari produk"
        }
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        promo: {context_text}
        client: {context_previous}
        Temukan 3 promo yang sesuai dengan info profile client. Ekstrak 3 nama produk bank BNI seperti nama tabungan, nama kartu kredit, nama kartu debit yang ada pada promo. Cari informasi dari produk tersebut. Outputkan hasil dalam format JSON seperti {json_format}."
        Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Output:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"carousel_data": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["carousel_data"] = re.sub(' +', ' ', output_stage["carousel_data"])
        output_stage["user_defined_type"] = "carousel"
        return output_stage
    

    async def watsonxai_product_reco_opt(self, context_previous):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=False)
        json_format = {
            "label": "nama produk BNI",
            "value": {
                "input": {
                    "text": "nama produk BNI, bisa berupa nama tabungan, kartu debit, atau kartu kredit"
                }
            }
        }
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        promo: {context_text}
        client: {context_previous}
        Temukan 3 promo yang sesuai dengan info profile client. Ekstrak 3 nama produk bank BNI seperti nama tabungan, nama kartu kredit, nama kartu debit yang ada pada promo. Cari informasi dari produk tersebut. Outputkan hasil dalam format JSON seperti {json_format}."
        Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Output:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"options": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["options"] = re.sub(' +', ' ', output_stage["options"])
        output_stage["options"] = ast.literal_eval(output_stage['options'])
        output_stage['title'] = "Pilih salah satu rekomendasi produk berikut:"
        output_stage['description'] = "opsi rekomendasi"
        output_stage['response_type'] = "option"

        return {"output": [output_stage]}
    
    
    async def watsonxai_product_promo(self, context_previous):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=True)
        json_format = {
            "alt": "nama produk BNI",
            "url": "jpg url jika ada",
            "title": "nama produk BNI, bisa berupa nama tabungan, kartu debit, atau kartu kredit",
            "description": "deskripsi dan persyaratan dari produk"
        }
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        promo: {context_text}
        client: {context_previous}
        Temukan 3 promo yang sesuai dengan info profile client. Ekstrak 3 nama produk bank BNI seperti nama tabungan, nama kartu kredit, nama kartu debit yang ada pada promo. Cari informasi dari produk tersebut. Outputkan hasil dalam format JSON seperti {json_format}."
        Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Output:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        output_stage["output"] = ast.literal_eval(output_stage['output'])
        return output_stage
    

    async def watsonxai_product_promo_crsl(self, context_previous):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=True)
        json_format = {
            "alt": "nama produk BNI",
            "url": "jpg url jika ada",
            "title": "nama produk BNI, bisa berupa nama tabungan, kartu debit, atau kartu kredit",
            "description": "deskripsi dan persyaratan dari produk"
        }
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        promo: {context_text}
        client: {context_previous}
        Temukan 3 promo yang sesuai dengan info profile client. Ekstrak 3 nama produk bank BNI seperti nama tabungan, nama kartu kredit, nama kartu debit yang ada pada promo. Cari informasi dari produk tersebut. Outputkan hasil dalam format JSON seperti {json_format}."
        Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Output:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"carousel_data": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["carousel_data"] = re.sub(' +', ' ', output_stage["carousel_data"])
        output_stage["user_defined_type"] = "carousel"
        return output_stage
    

    async def watsonxai_product_promo_opt(self, context_previous):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(context_previous, PROJECT_ID, text_list=True)
        json_format = {
            "label": "nama produk BNI",
            "value": {
                "input": {
                    "text": "nama produk BNI, bisa berupa nama tabungan, kartu debit, atau kartu kredit"
                }
            }
        }
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        promo: {context_text}
        client: {context_previous}
        Temukan 3 promo yang sesuai dengan info profile client. Ekstrak 3 nama produk bank BNI seperti nama tabungan, nama kartu kredit, nama kartu debit yang ada pada promo. Cari informasi dari produk tersebut. Outputkan hasil dalam format JSON seperti {json_format}."
        Jangan menambahkan kesimpulan, keterangan, duplikasi jawaban, dan informasi tambahan selain dari json format yang diminta.
        Output:"""
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"options": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["options"] = re.sub(' +', ' ', output_stage["options"])
        output_stage["options"] = ast.literal_eval(output_stage['options'])
        output_stage['title'] = "Pilih salah satu rekomendasi produk berikut:"
        output_stage['description'] = "opsi rekomendasi"
        output_stage['response_type'] = "option"

        return {"output": [output_stage]}

    async def watsonxai_promo_information(self, product, user_question):
        PROJECT_ID = self.WD_PROJECT_ID
        context_text = self.send_to_watsondiscovery(product, PROJECT_ID, text_list=False)

        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        Konteks_wd: {context_text}
        Pertanyaan: {user_question}
        Jawab pertanyaan dari Konteks_wd mengenai promo BNI. Jawaban dalam bentuk satu paragraf dengan maksimum 5 kalimat. 
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        return output_stage

    async def watsonxai_product_information(self, product, user_question):
        PROJECT_ID = self.WD_PROJECT_ID_2
        context_text = self.send_to_watsondiscovery(product, PROJECT_ID, text_list=False)

        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        Konteks_wd: {context_text}
        Pertanyaan: {user_question}
        Jawab pertanyaan dari Konteks_wd mengenai produk BNI. Jawaban dalam bentuk satu paragraf dengan maksimum 5 kalimat. 
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])
        return output_stage
    
    async def watsonxai_product_summary(self, product):
        PROJECT_ID = self.WD_PROJECT_ID_3
        context_text = self.send_to_watsondiscovery(product, PROJECT_ID, text_list=True)

        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        context_text: {context_text}
        produk: {product}
        Tolong rangkum dengan detail "produk" di dalam "context_text" yang menjelaskan tujuan, manfaat, fasilitas, syarat, biaya dari "produk". Jawaban tidak boleh lebih dari 8 kalimat dan harus kedalam bentuk satu paragraf.
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])

        return output_stage
    
    async def watsonxai_product_comparison(self, product_A, product_B):
 
        prompt_stage = f"""Kamu adalah asisten yang membantu, menghormati, dan jujur. Selalu jawab sebisa mungkin, sambil tetap aman. Jawaban Anda tidak boleh mengandung konten yang berbahaya, tidak etis, rasial, seksis, beracun, berbahaya, atau ilegal. Pastikan bahwa respons Anda tidak memihak dan bersifat positif.
        kartu_{product_A}: {product_A}
        kartu_{product_B}: {product_B}
        Tolong bandingkan dan dibuat dalam nomor urut berupa tujuan, fasilitas, syarat, biaya dan manfaat dari kartu "kartu_{product_A}" dan kartu "kartu_{product_B}".
        Jawaban:
        """
        output_stage = self.send_to_watsonxai(prompts=[prompt_stage], stop_sequences=[])
        output_stage = {"output": str(output_stage.strip()).replace('\n\n', ' ').replace('*', '<li>')}
        output_stage["output"] = re.sub(' +', ' ', output_stage["output"])

        return output_stage
