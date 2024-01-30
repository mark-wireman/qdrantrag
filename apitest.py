import requests
import json
import argparse

QUESTION = ""
MODEL = ""
URL = ""

def get_values():
    parser = argparse.ArgumentParser()
    parser.add_argument("url",help="The URL to the Ollama server.",type=ascii)
    parser.add_argument("model",help="The LLM model to use.",type=ascii)
    parser.add_argument("question",help="Question to ask",type=ascii)
    args = parser.parse_args()

    URL = str(args.url)
    MODEL = str(args.model)
    QUESTION = str(args.question)

    print(URL+"\t"+MODEL+"\t"+QUESTION+"\n")

url = "http://192.168.50.136:11434/api/generate"
question = "how to apply the ooda loop to secure cloud modernization?"
model = "mixtral"

data = {"model": model,"prompt": question,"stream": False}
response = requests.request("POST", url, json=data)
#get_values()
#url = "" + URL
#post_request = {"model": MODEL,"prompt": QUESTION,"stream": False}
#response = requests.request("POST", URL, json=post_request)
data = response.json()

response_string = data["response"]

print(response_string)
