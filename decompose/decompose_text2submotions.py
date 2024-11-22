from pathlib import Path
import os
from os import listdir
from dotenv import load_dotenv
import json
from typing_extensions import override

import hydra
from openai import AssistantEventHandler, OpenAI
from openai import OpenAI


client = OpenAI()

def parser_json_MCD(answers, end=10):
    
    filtered_answer = answers
    bad_characters = ["json", "`", "\n"]
    for b in bad_characters:
        filtered_answer = filtered_answer.replace(b, "").strip()
    
    parsed_answer = json.loads(filtered_answer)

    # Check that end is less than start for each submovement
    for submov in parsed_answer["decomposition"]:
        submov["end"] = submov["end"] if submov["end"] < end else end
        if not submov["start"] < submov["end"]:
            print("Start is greater than end")
            assert submov["start"] < submov["end"]

    # Check that at least one submovement starts from 0
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
        parsed_answer["decomposition"][min_id]["start"] = 0

    # Check that at least one submovement ends at original end time
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
        parsed_answer["decomposition"][max_id]["end"] = end

    return parsed_answer


def intersection(a, b):
    if b["start"] < a["start"]:
        a, b = b, a
    if a["end"] > b["start"]:
        return True
    return False


def parser_json_STMC(answers, end=10):
    # Parser with more constraits: check of the body part names and check of time interlap of body parts  
    filtered_answer = answers
    bad_characters = ["json", "`", "\n"]
    for b in bad_characters:
        filtered_answer = filtered_answer.replace(b, "").strip()
    
    parsed_answer = json.loads(filtered_answer)

    # Check that end is less than start for each submovement
    for submov in parsed_answer["decomposition"]:
        submov["end"] = submov["end"] if submov["end"] < end else end
        if not submov["start"] < submov["end"]:
            print("Start is greater than end")
            assert submov["start"] < submov["end"]

    # Check that at least one submovement starts from 0
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
        parsed_answer["decomposition"][min_id]["start"] = 0

    # Check that at least one submovement ends at original end time
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
        print("Max not found!")
        parsed_answer["decomposition"][max_id]["end"] = end

    # check if interlapping of body parts
    num_submovmenets = len(parsed_answer["decomposition"])
    for i in range(0, num_submovmenets-1):
        for j in range(i+1, num_submovmenets):
            inters = intersection(parsed_answer["decomposition"][i], parsed_answer["decomposition"][j])
            if inters:
                inters_body_parts = set(parsed_answer["decomposition"][i]["body parts"]).intersection(parsed_answer["decomposition"][j]["body parts"])
                if len(inters_body_parts) > 0:
                    print("Interlapping body parts")
                    raise ValueError("Interlapping body parts")
    
    # check if body parts are right
    for e, submov in enumerate(parsed_answer["decomposition"]):
        bp = submov["body parts"]
        for b in bp:
            if b not in ["head", "spine", "left arm", "right arm", "legs"]:
                print("Wrong body part")
                raise ValueError("Wrong body part")

    return parsed_answer


def invia_richiesta(message_text, thread, assistant):

    ### CREATE MESSAGES 
    thread_message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"{message_text}", 
    )
    print("Running assistant...")
    answer = None
    event_handler = EventHandler(answer)
    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()
    
    answer = event_handler.answer
    return answer


def decompose(text, assistant, parser=parser_json_MCD):
    thread = client.beta.threads.create()
    
    # Loop attraverso la lista di messaggi
    answer = invia_richiesta([text], thread=thread, assistant=assistant)
    
    end_original = float(text.split("#")[2].strip()) 
    end_original = end_original if end_original < 10 else 10      
    
    parsed_answers = parser(answer, end_original)

    print(f"Decomposed output:")
    print(parsed_answers)
    
    return parsed_answers


class EventHandler(AssistantEventHandler):

    def __init__(self, answer):
        super().__init__()
        self.answer = answer

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

        self.answer = message_content.value


# Load the .env file
load_dotenv() 
# Create the client using the API key
client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'),)
# Calcolo del percorso assoluto per la directory "configs"
script_dir = os.path.dirname(__file__)  # Directory dello script
config_path = os.path.abspath(os.path.join(script_dir, "..", "configs"))


@hydra.main(config_path=config_path, config_name="decompose", version_base="1.3")
def main(cfg):
    dataset = cfg.dataset # "kitml" # "humanml3d"
    split = cfg.split # "complex"
    gpt_model = "gpt-4o-mini"
    compose = cfg.compose # "MCD" or "STMC"
    only_not_generated = cfg.only_not_generated # True or False

    if compose == "MCD":
        instructions_path = "decompose/istructions/MCD_instruction.txt"
        save_path = f"datasets/annotations/{dataset}/splits/{split}/submotions/"
        examples_path = f"decompose/examples/MCD_examples_{dataset}.txt"
        parser = parser_json_MCD 
    else:
        instructions_path = "decompose/istructions/STMC_instruction.txt"
        save_path = f"datasets/annotations/{dataset}/splits/{split}/submotions_stmc/"
        examples_path = f"decompose/examples/STMC_examples_{dataset}.txt"
        parser = parser_json_STMC

    ids_path = f'datasets/annotations/{dataset}/splits/{split}/test.txt'
    train_file_path = f"datasets/annotations/{dataset}/splits/{split}/gpt_train_texts.txt"
    
    # Load the test ids
    with open(ids_path, 'r') as f:
        test_ids = f.readlines()
    test_ids = [id.strip() for id in test_ids]
    # Load istructions file
    with open(instructions_path, 'r') as f:
        instructions = f.read()
    
    # Just to debug, if DEBUG > 0 it will consider only the first #DEBUG samples  
    DEBUG = 0
    if DEBUG != 0:
        test_ids = test_ids[:DEBUG]
    # Check if decomposed files already exists
    existing_files = listdir(save_path)
    existing_ids = [f"{Path(t).stem}" for t in existing_files]
    ids_2_augment = []
    for id in test_ids:
        if id not in existing_ids:
            ids_2_augment.append(id)

    if only_not_generated:
        test_ids = ids_2_augment

    print(f"Elaborating the following {len(test_ids)} samples: {test_ids}")
    if len(test_ids) == 0:
        print("No id to elaborate - all files already generated")
        exit() 

    # CREATE TRAIN FILE FOR GPT
    annotations_train_path = f"{os.getcwd()}/datasets/annotations/{dataset}/splits/{split}/annotations_train.json" 
    annotations_train = json.load(open(annotations_train_path))
    train_samples = []
    ids_train_path = Path(ids_path).parents[0] / "train.txt"
    with open(ids_train_path, 'r') as f:
        train_ids = f.readlines()
    train_ids = [id.strip() for id in train_ids]
    for idx in train_ids:
        tot_ann = annotations_train[idx]
        for ann in tot_ann["annotations"]:
            duration = round(ann["end"] - ann["start"])
            if duration > 10:
                continue
            m = ann["text"] + " # " + str(0) + " # " + str(duration)
            train_samples.append(m)
    print(f"Collected {len(train_samples)} samples for training file")
    with open(train_file_path, 'w') as outfile:
        for m in train_samples:
            outfile.write(m + '\n')

    ### CREATE LIST OF TEST MESSAGES
    messages = []
    annotations_test_path = f"{os.getcwd()}/datasets/annotations/{dataset}/splits/{split}/annotations_test.json" 
    annotations_test = json.load(open(annotations_test_path))
    for idx in test_ids:
        ann = annotations_test[idx]["annotations"][0] 
        m = ann["text"] + " # " + str(0) + " # " + str(ann["end"] - ann["start"])
        messages.append(m)

    ### CREATE ASSISTENT
    assistant = client.beta.assistants.create(
        model=gpt_model,
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
    # Print the status and the file counts of the batch to see the result of this operation.
    # print(file_batch.status)
    # print(file_batch.file_counts)

    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    errors = []
    for j, m in enumerate(messages):

        idx = test_ids[j].strip()
        print(f"File {idx}: - {j}/{len(messages)}")
        print(f"### Input \n{m}")
        try:
            parsed_answers = decompose(m, assistant=assistant, parser=parser)
        except Exception as e:
            print(f"Error: {e}")
            errors.append(idx)
            continue

        # Save the decomposed file
        with open(f"{save_path}/{idx}.txt", 'w') as f:
            for submov in parsed_answers["decomposition"]:
                text = submov["text"]
                start = submov["start"]
                end = submov["end"]
                f.write(f"{text} # {start} # {end} # spine\n")

    print(f"Errors {len(errors)}: {errors}")


if __name__ == "__main__":
    main()