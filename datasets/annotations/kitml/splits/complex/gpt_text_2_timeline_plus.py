from pathlib import Path
from openai import OpenAI
import requests
import os
from os import listdir
from dotenv import load_dotenv
import json

from typing_extensions import override
from openai import AssistantEventHandler, OpenAI

client = OpenAI()

def parse_answers_json(answers, end=10):
    
    filtered_answer = answers
    bad_characters = ["json", "`", "\n"]
    for b in bad_characters:
        filtered_answer = filtered_answer.replace(b, "").strip()
    
    parsed_answer = json.loads(filtered_answer)

    # Controllo che start < end, che end <= 10
    for submov in parsed_answer["decomposition"]:
        submov["end"] = submov["end"] if submov["end"] < end else end
        if not submov["start"] < submov["end"]:
            print("Start is greater than end")
            return None
            assert submov["start"] < submov["end"]

    # controllo che uno inizi da 0
    min_, min_id = 1000, None
    find_zero = False
    for e, submov in enumerate(parsed_answer["decomposition"]):
        if submov["start"] < min_:
            min_ = submov["start"]
            min_id = e
        if submov["start"] == 0:
            find_zero = True
            break
    if not find_zero:
        print("Zero not found!")
        parsed_answer["decomposition"][min_id]["start"] = 0

    # controllo che uno inizi finisca a max end
    max_, max_id = -1, None
    find_top = False
    for e, submov in enumerate(parsed_answer["decomposition"]):
        if submov["end"] > max_:
            max_ = submov["end"]
            max_id = e
        if submov["end"] == end:
            find_top = True
            break
    if not find_top:
        print("Zero not found!")
        parsed_answer["decomposition"][max_id]["end"] = end

    return parsed_answer

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"assistant > ", end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"assistant > {tool_call.type}\n", flush=True)

    @override
    def on_message_done(self, message) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        answers.append(message_content.value)

# Carica le variabili d'ambiente
load_dotenv() 

# Imposta la tua chiave API di OpenAI
client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'),)

def invia_richiesta(messagge_texts):

    ### CREATE MESSAGES 
    for message_text in messagge_texts:
        # print(f"Sending message: {message_text}")
        thread_message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"{message_text}", ### DEBUG
        )
    print("Running assistant...")
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


dataset = "kitml" # "kitml" # "humanml3d"
instructions_path = "neverseen_dataset/istructions/gpt_istructions_times.txt"
examples_path = "neverseen_dataset/examples/gpt_examples_times_kitml.txt"
save_path = f"datasets/annotations/{dataset}/splits/me/timelines/"
ids_path = f'datasets/annotations/{dataset}/splits/me/test.txt'
ids_path = f'datasets/annotations/{dataset}/splits/me/test_to_fix.txt'
train_file_path = f"datasets/annotations/{dataset}/splits/me/gpt_train_texts.txt"

DEBUG = 0

with open(ids_path, 'r') as f:
    test_ids = f.readlines()
test_ids = [id.strip() for id in test_ids]
with open(instructions_path, 'r') as f:
    instructions = f.read()

if DEBUG != 0:
    test_ids = test_ids[:DEBUG]

existing_files = listdir(save_path)
existing_ids = [f"{Path(t).stem}" for t in existing_files]
ids_2_augment = []
for id in test_ids:
    if id not in existing_ids:
        ids_2_augment.append(id)

test_ids = ids_2_augment

print(f"Elaborating the following {len(test_ids)} samples: {test_ids}")
if len(test_ids) == 0:
    print("No id to elaborate")
    exit() 

# CREATE TRAIN FILE FOR GPT
annotations_train_path = f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{dataset}/splits/me/annotations_train.json" 
annotations_train = json.load(open(annotations_train_path))
train_samples = []
ids_train_path = Path(ids_path).parents[0] / "train.txt"
with open(ids_train_path, 'r') as f:
    train_ids = f.readlines()
train_ids = [id.strip() for id in train_ids]
for idx in train_ids:
    tot_ann = annotations_train[idx]
    for ann in tot_ann["annotations"]:
        m = ann["text"] + " # " + str(0) + " # " + str(ann["end"] - ann["start"])
        train_samples.append(m)
print(f"Collected {len(train_samples)} samples for training file")
with open(train_file_path, 'w') as outfile:
    # Scrivi solo le righe pari (indice 1, 3, 5, ...) nel file di output
    for m in train_samples:
        outfile.write(m + '\n')

### CREATE LIST OF TEST MESSAGES
messages = []
annotations_test_path = f"/andromeda/personal/lmandelli/stmc/datasets/annotations/{dataset}/splits/me/annotations_test.json" 
annotations_test = json.load(open(annotations_test_path))
for idx in test_ids:
    ann = annotations_test[idx]["annotations"][0] 
    # Prendiamo solo la prima annotazione
    m = ann["text"] + " # " + str(0) + " # " + str(ann["end"] - ann["start"])
    messages.append(m)

### CREATE ASSISTENT
assistant = client.beta.assistants.create(
    model="gpt-4o-mini",#gpt-4o-mini
    instructions=instructions,
    name="Text to timeline assistant",
    tools=[{"type": "file_search"}],
    temperature=0.0,
)

### UPLOAD FILE
# Create a vector store
vector_store = client.beta.vector_stores.create(name="texts_train")
# Ready the files for upload to OpenAI
file_paths = [train_file_path, examples_path]
file_streams = [open(path, "rb") for path in file_paths]
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
print("Uploading file...")
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)
# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)
### CREATE THREAD
errors = []
for j, m in enumerate(messages):
    thread = client.beta.threads.create()

    idx = test_ids[j].strip()
    print(f"File {idx}: - {j}/{len(messages)}")
    print(f"### Input \n{m}")
    answers=[]
    # Loop attraverso la lista di messaggi
    invia_richiesta([m])
    end_original = float(m.split("#")[2].strip()) 
    end_original = end_original if end_original < 10 else 10      
    parsed_answers = parse_answers_json(answers[0], end_original)
    if parsed_answers is None:
        errors.append(idx)
        continue

    print(f"### Output")
    print(parsed_answers)
        
    # Salviamo la timeline
    with open(f"{save_path}/{idx}.txt", 'w') as f:
        for submov in parsed_answers["decomposition"]:
            text = submov["text"]
            start = submov["start"]
            end = submov["end"]
            f.write(f"{text} # {start} # {end} # spine\n")

print(f"Errors {len(errors)}: {errors}")