# DistillLama

Distill knowledge with local LLMs


![meme](assets/drinking_lama.jpeg)



Distilllama let you distill your knowledge intro your locally running LLM, using the same model for embeddings. 

Based on LangChain and LLama.cpp This work is heavily in progress.


Install the requirements:
```shell
pip install -r requirements.txt
```

Copy the sample.env to .env file:
```shell
cp sample.env .env
```

Edit the .env file setting the variables according to your path.

```shell
PERSIST_DIRECTORY=./db
MODEL_TYPE=LlamaCpp
MODEL_PATH=YOUR_MODEL_FILE.bin
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
HF_EMB=True
MODEL_N_CTX=2048
```

There are several models suited for this, one that has good performance/resources ratio so far is:

https://huggingface.co/TheBloke/wizard-mega-13B-GGML

Any other model with the ggml format from the latest release of llama.cpp will work as well, each model card has the requirements for RAM and disk size, so you can choose one accordingly to your local resources.


Run it:
```shell
python3 distilllama.py
```

On the first run, distilllama will load all the text documents found in the documents folder.
All ingested documents are moved into the processed subfolder.

New information can also be provided from the chat interface by typing 'Information'.
