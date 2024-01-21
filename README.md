# On-Boarding and Help Center for API user

We have developed a dynamic Python script hosted on IBM Cloud Code Engine to enhance user interaction by enabling inquiries. This script seamlessly integrates various cutting-edge technologies, including 
- Watson Assistant,
- Watson Discovery,
- watsonx.ai,
- Code Engine, and
- Cloud Object Storage. 

The user's questions trigger a Language Model (LLM), which extracts pertinent information from our knowledge base, primarily sourced from Watson Discovery. The LLM crafts engaging responses based on the knowledge gleaned, creating an interactive and informative user experience. The orchestration of these advanced technologies ensures a sophisticated and responsive system, harnessing the capabilities of IBM Cloud for an enriched conversational interaction.

## The Expected Result
![test-bni-gif](https://github.com/Client-Engineering-Indonesia/help-center/assets/32385413/e64c40a9-313e-4e3f-b8fd-dc904112787d)

## The Architecture
<img width="1424" alt="image" src="https://github.com/Client-Engineering-Indonesia/help-center/assets/32385413/ceb4d28c-70e0-4cc0-965c-4a7be6d13a82">


## Required Credentials:
- watsonx.ai project ID
- IAM APIKEY
- Cloud Object Storage APIKEY, INSTANCE_CRN, URL Public Endpoint
- Discovery API key and Project ID

## Running Locally using Docker
```
git clone https://github.com/Client-Engineering-Indonesia/help-center

cd help-center

docker build -t imagetagname .

docker run \
--env "COS_APIKEY=your-secret-value" \
--env "COS_APIKEY_PUB=your-secret-value" \
--env "COS_INSTANCE_CRN=your-secret-value" \
--env "COS_INSTANCE_CRN_PUB=your-secret-value" \
--env "endpoint_url_private=your-secret-value" \
--env "endpoint_url_public=your-secret-value" \
--env "WX_API_KEY=your-secret-value" \
--env "WX_PROJECT_ID=your-secret-value" \
--env "WX_URL=your-secret-value" \
--env "WD_PROJECT_ID=your-secret-value" \
--env "WD_API_KEY=your-secret-value" \
--env "WD_URL=your-secret-value" \
-p 8080:8080 --name your-container-name your-image-name
```


## Deployment using Code Engine
The detail step by step deployment using code engine please visit this [medium]()


### Reference
- To learn further about Cloud Object Storage, visit this medium: https://medium.com/@thursysatriani/managing-data-in-ibm-cloud-66a3c44e2b8c
- To learn further about Docker Container, visit this medium: https://medium.com/@thursysatriani/docker-desktop-cli-essentials-796f46feb4c3
