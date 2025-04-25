import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
#from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
#from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


cosi = torch.nn.CosineSimilarity(dim=0)

binomial_data = pd.read_csv('../Data/all_sentences.csv')
sentences=binomial_data['Sentence']
word1 = binomial_data['Word1']
word2 = binomial_data['Word2']

device = 'cpu'

def load_model(model_name, device = 'cpu', checkpoint=None): #let's test olmo, gpt, and llama
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision = checkpoint)
    model.config.output_hidden_states = True
    model.config.pad_token_id = model.config.eos_token_id
    model = model.to(device)
    return model, tokenizer 



def get_semantic_representation_batch(model, tokenizer, input_texts, word1, word2, layer, type='v1'):
    # Tokenize the input texts with offsets for locating words
    inputs = tokenizer(input_texts, padding=True, return_tensors='pt', return_offsets_mapping=True)
    input_ids = inputs.input_ids.to(device)
    offset_mappings = inputs.offset_mapping
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
    phrase_embeddings_batch_list = []
    # Process each sentence and its respective word pairs
    for i, (input_text, w1, w2) in enumerate(zip(input_texts, word1, word2)):
        token_vecs = hidden_states[i]
        offsets = offset_mappings[i]   
        # Find the start and end character indices of w1 and w2
        w1_start = input_text.find(w1)
        if type=='v2':
            w1_start = w1_start - 1 
        w1_end = w1_start + len(w1)
        w2_start = input_text.find(w2)
        w2_end = w2_start + len(w2)
        #print(input_text, w1, w2, w1_start, w2_start, offsets) #for debugging
        # Locate the token indices that correspond to w1 and w2
        start_index, end_index = None, None
        for j, (start, end) in enumerate(offsets):
            # Find the start index for w1
            if start_index is None and start == w1_start:
                start_index = j
            # Find the end index for w2
            if end_index is None and end == w2_end:
                end_index = j
            # If both indices are found, break out of the loop
            if start_index is not None and end_index is not None:
                break
        # Calculate the phrase embedding if both indices are found
        if start_index is not None and end_index is not None:
            phrase_embedding_batch = torch.mean(token_vecs[start_index:end_index + 1, :], dim=0)
            phrase_embeddings_batch_list.append(phrase_embedding_batch)
        else:
            print(f"Could not locate tokens for '{w1}' or '{w2}' in input_text '{input_text}'. Skipping this pair.")
    # Convert the list of embeddings to a single tensor if it's not empty
    if phrase_embeddings_batch_list:
        phrase_embeddings_batch_tensor = torch.stack(phrase_embeddings_batch_list)
    else:
        phrase_embeddings_batch_tensor = None
    return phrase_embeddings_batch_tensor

  
def get_compositional_semantic_embeddings(model, tokenizer, multi_word_phrase, layer): #this will take in a phrase (as a list) and return its compositional representation (returns the representations of each individual word then takes the mean of those representations)
  inputs = tokenizer(multi_word_phrase, padding=True, return_tensors='pt', return_offsets_mapping=True).to(device)
  input_ids = inputs.input_ids
  offset_mapping = inputs.offset_mapping
  attention_mask = inputs.attention_mask
  with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[layer]
  word_embeddings = []
  for i, word in enumerate(multi_word_phrase):
    word_offsets = offset_mapping[i].tolist()
    token_embeddings = hidden_states[i]
    word_token_embeddings = []
    for j, (start, end) in enumerate(word_offsets):
      if attention_mask[i][j] == 1 and start != end:  # skip padding and non-token parts
        word_token_embeddings.append(token_embeddings[j])
    if word_token_embeddings:
      word_embedding = torch.mean(torch.stack(word_token_embeddings), dim=0)
      word_embeddings.append(word_embedding) 
  combined_embedding = torch.mean(torch.stack(word_embeddings),dim=0)
  return [combined_embedding]


def main(model_name, sentences, word1, word2, layer = -2, checkpoint = None):
    print(f"Processing {model_name} now.")
    model, tokenizer = load_model(model_name, checkpoint=checkpoint)
    n_batches = len(sentences) / 1 #
    model_name_for_saving = model_name.replace("/", "_")
    input_texts_sentences = np.array_split(sentences, n_batches)
    input_texts_sentences = [x.tolist() for x in [*input_texts_sentences]]

    input_texts_word1 = np.array_split(word1, n_batches)
    input_texts_word1 = [x.tolist() for x in [*input_texts_word1]]

    input_texts_word2 = np.array_split(word2, n_batches)
    input_texts_word2 = [x.tolist() for x in [*input_texts_word2]]


    sentence_binom_order = [[]]
    batch_sentences = [[]]
    timer = 0

    #the way OlmO counts the offsets is different for some reason. So we have to subtract one from the index when it's not 
    #an olmo model. It's a bit weird but 
    if model_name == 'gpt2-xl' or model_name == 'meta-llama/Llama-2-7b-hf' or model_name == 'gpt2':
        modeltype = 'v2'
    else:
        modeltype = 'v1'

    for minibatch, word1, word2 in zip(input_texts_sentences, input_texts_word1, input_texts_word2):
        #assert len(input_texts_sentences) == len(input_texts_word1) == len(input_texts_word2), "Input lists must be the same length"
        timer += 1
        if timer % 100 == 0:
            print(timer)
        if len(word1) != len(word2):
            raise ValueError("word1 and word2 lists must be the same length")
        binom = [word1[i] + ' and ' + word2[i] for i in range(len(word1))]
        batch_run = get_semantic_representation_batch(model, tokenizer, minibatch, word1, word2, type=modeltype, layer = layer)
        batch_sentences.extend(batch_run)
        #sentence_binom_order.extend(binom)

    batch_sentences = batch_sentences[1:]
    sentence_representations = torch.stack(batch_sentences)
    sentence_binom_order = sentence_binom_order[1:]

    binom_order = [[]]
    compositional_representations = [[]]
    timer = 0
    for word1, word2 in zip(input_texts_word1, input_texts_word2):
        phrase = [word1[i] + ' and ' + word2[i] for i in range(len(word1))]
        timer += 1

        if timer % 100 == 0:
            print(timer)
        
        # Get the compositional semantic embeddings for the current batch
        batch_run_binoms = get_compositional_semantic_embeddings(model, tokenizer, phrase, layer=layer)
        
        # Process the embeddings
        compositional_representations.extend(batch_run_binoms)
        
        # Extend the binom order with the current phrase
        binom_order.extend(phrase)

    binom_order = binom_order[1:]
    compositional_representations = torch.stack(compositional_representations[1:], dim=0)


    cosine_diffs_comp = [[]]

    compositional_representations.shape


    for rep1, rep2 in zip(sentence_representations, compositional_representations):
    
        cosine_similarities = cosi(rep1, rep2)
        #print(cosine_similarities)
        cosine_similarities = cosine_similarities.item()
        
        cosine_diffs_comp.append(cosine_similarities)
    
    cosine_diffs_comp = cosine_diffs_comp[1:]

    cosine_diffs_comp_df = pd.DataFrame({'cosine_diffs': cosine_diffs_comp})

    cosine_diffs_comp_df['binom'] = binom_order

    
    cosine_diffs_comp_df.to_csv(f"../Data/{model_name_for_saving}_{checkpoint}_layer_{layer}_compositional_cosine_diffs.csv")
    print(f"Finished Processing {model_name} now.")


def main2(model_name, sentences, word1, word2, checkpoints, layers = [-2]):
    n_batches = len(sentences) / 1 #
    model_name_for_saving = model_name.replace("/", "_")
    input_texts_sentences = np.array_split(sentences, n_batches)
    input_texts_sentences = [x.tolist() for x in [*input_texts_sentences]]

    input_texts_word1 = np.array_split(word1, n_batches)
    input_texts_word1 = [x.tolist() for x in [*input_texts_word1]]

    input_texts_word2 = np.array_split(word2, n_batches)
    input_texts_word2 = [x.tolist() for x in [*input_texts_word2]]

    for checkpoint in checkpoints:
        model, tokenizer = load_model(model_name, checkpoint=checkpoint)
        
        for layer in layers:
            print(f"Processing {model_name} at {checkpoint} at layer {layer} now.")
            layer = layer
            sentence_binom_order = [[]]
            batch_sentences = [[]]
            timer = 0

            #the way OlmO counts the offsets is different for some reason. So we have to subtract one from the index when it's not 
            #an olmo model. It's a bit weird but 
            if model_name == 'gpt2-xl' or model_name == 'meta-llama/Llama-2-7b-hf' or model_name == 'gpt2':
                modeltype = 'v2'
            else:
                modeltype = 'v1'

            for minibatch, word1, word2 in zip(input_texts_sentences, input_texts_word1, input_texts_word2):
                #assert len(input_texts_sentences) == len(input_texts_word1) == len(input_texts_word2), "Input lists must be the same length"
                timer += 1
                if timer % 100 == 0:
                    print(timer)
                if len(word1) != len(word2):
                    raise ValueError("word1 and word2 lists must be the same length")
                binom = [word1[i] + ' and ' + word2[i] for i in range(len(word1))]
                batch_run = get_semantic_representation_batch(model, tokenizer, minibatch, word1, word2, type=modeltype, layer=layer)
                batch_sentences.extend(batch_run)
                #sentence_binom_order.extend(binom)

            batch_sentences = batch_sentences[1:]
            sentence_representations = torch.stack(batch_sentences)
            sentence_binom_order = sentence_binom_order[1:]

            binom_order = [[]]
            compositional_representations = [[]]
            timer = 0
            for word1, word2 in zip(input_texts_word1, input_texts_word2):
                phrase = [word1[i] + ' and ' + word2[i] for i in range(len(word1))]
                timer += 1

                if timer % 100 == 0:
                    print(timer)
                
                # Get the compositional semantic embeddings for the current batch
                batch_run_binoms = get_compositional_semantic_embeddings(model, tokenizer, phrase, layer=layer)
                
                # Process the embeddings
                compositional_representations.extend(batch_run_binoms)
                
                # Extend the binom order with the current phrase
                binom_order.extend(phrase)

            binom_order = binom_order[1:]
            compositional_representations = torch.stack(compositional_representations[1:], dim=0)


            cosine_diffs_comp = [[]]

            compositional_representations.shape


            for rep1, rep2 in zip(sentence_representations, compositional_representations):
            
                cosine_similarities = cosi(rep1, rep2)
                #print(cosine_similarities)
                cosine_similarities = cosine_similarities.item()
                
                cosine_diffs_comp.append(cosine_similarities)
            
            cosine_diffs_comp = cosine_diffs_comp[1:]

            cosine_diffs_comp_df = pd.DataFrame({'cosine_diffs': cosine_diffs_comp})

            cosine_diffs_comp_df['binom'] = binom_order

            
            cosine_diffs_comp_df.to_csv(f"../Data/{model_name_for_saving}_{checkpoint}_layer_{layer}_compositional_cosine_diffs.csv")
            print(f"Finished Processing {model_name} at checkpoint {checkpoint} and layer {layer} now.")





def get_semantic_representation_batch_all_layers(model, tokenizer, input_texts, word1, word2, layers, type='v1'):
    # Tokenize the input texts with offsets for locating words
    inputs = tokenizer(input_texts, padding=True, return_tensors='pt', return_offsets_mapping=True)
    input_ids = inputs.input_ids.to(device)
    offset_mappings = inputs.offset_mapping
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states_all = outputs.hidden_states
    embeddings_all_layers = {}
    for layer in layers:
        hidden_states = hidden_states_all[layer]
        phrase_embeddings_batch_list = []
        # Process each sentence and its respective word pairs
        for i, (input_text, w1, w2) in enumerate(zip(input_texts, word1, word2)):
            token_vecs = hidden_states[i]
            offsets = offset_mappings[i]   
            # Find the start and end character indices of w1 and w2
            w1_start = input_text.find(w1)
            if type=='v2':
                w1_start = w1_start - 1 
            w1_end = w1_start + len(w1)
            w2_start = input_text.find(w2)
            w2_end = w2_start + len(w2)
            #print(input_text, w1, w2, w1_start, w2_start, offsets) #for debugging
            # Locate the token indices that correspond to w1 and w2
            start_index, end_index = None, None
            for j, (start, end) in enumerate(offsets):
                # Find the start index for w1
                if start_index is None and start == w1_start:
                    start_index = j
                # Find the end index for w2
                if end_index is None and end == w2_end:
                    end_index = j
                # If both indices are found, break out of the loop
                if start_index is not None and end_index is not None:
                    break
            # Calculate the phrase embedding if both indices are found
            if start_index is not None and end_index is not None:
                phrase_embedding_batch = torch.mean(token_vecs[start_index:end_index + 1, :], dim=0)
                phrase_embeddings_batch_list.append(phrase_embedding_batch)
            else:
                print(f"Could not locate tokens for '{w1}' or '{w2}' in input_text '{input_text}'. Skipping this pair.")
        # Convert the list of embeddings to a single tensor if it's not empty
        if phrase_embeddings_batch_list:
            phrase_embeddings_batch_tensor = torch.stack(phrase_embeddings_batch_list, dim=0)
        else:
            phrase_embeddings_batch_tensor = None
        embeddings_all_layers[layer] = phrase_embeddings_batch_tensor
    return embeddings_all_layers

  
def get_compositional_semantic_embeddings_all_layers(model, tokenizer, multi_word_phrase, layers): #this will take in a phrase (as a list) and return its compositional representation (returns the representations of each individual word then takes the mean of those representations)
  inputs = tokenizer(multi_word_phrase, padding=True, return_tensors='pt', return_offsets_mapping=True).to(device)
  input_ids = inputs.input_ids
  offset_mapping = inputs.offset_mapping
  attention_mask = inputs.attention_mask
  with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states_all = outputs.hidden_states
  embeddings_all_layers = {}
  for layer in layers:
    hidden_states = hidden_states_all[layer]
    word_embeddings = []
    for i, word in enumerate(multi_word_phrase):
        word_offsets = offset_mapping[i].tolist()
        token_embeddings = hidden_states[i]
        word_token_embeddings = []
        for j, (start, end) in enumerate(word_offsets):
            if attention_mask[i][j] == 1 and start != end:  # skip padding and non-token parts
                word_token_embeddings.append(token_embeddings[j])
        if word_token_embeddings:
            word_embedding = torch.mean(torch.stack(word_token_embeddings), dim=0)
            word_embeddings.append(word_embedding) 
    combined_embedding = torch.mean(torch.stack(word_embeddings),dim=0)
    embeddings_all_layers[layer] = combined_embedding
  return embeddings_all_layers


def main_multiple_layers(model_name, sentences, word1, word2, checkpoints, layers=[-2]):
    import numpy as np
    import pandas as pd
    import torch
    
    n_batches = int(len(sentences) / 1)  # Ensure integer division
    model_name_for_saving = model_name.replace("/", "_")
    
    def split_and_prepare(data):
        split_data = np.array_split(data, n_batches)
        return [x.tolist() for x in split_data]
    
    input_texts_sentences = split_and_prepare(sentences)
    input_texts_word1 = split_and_prepare(word1)
    input_texts_word2 = split_and_prepare(word2)
    
    if checkpoints:
        for checkpoint in checkpoints:
            layers_key = {f"layer{i+1}": value for i, value in enumerate(layers)}
            layers_names = {key: [] for key in layers_key.keys()}
            
            model, tokenizer = load_model(model_name, checkpoint=checkpoint)
            
            print(f"Processing {model_name} at {checkpoint} now.")
            
            # Determine model type
            if model_name in ['gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'gpt2']:
                modeltype = 'v2'
            else:
                modeltype = 'v1'
            

            comp_layers_names = {key: [] for key in layers_key.keys()}
            binom_order = []
                    
            # Process compositional embeddings
            timer = 0
            for w1, w2 in zip(input_texts_word1, input_texts_word2):
                timer += 1
                if timer % 100 == 0:
                        print(timer)
                phrase = [f"{w1[i]} and {w2[i]}" for i in range(len(w1))]
                binom_order.extend(phrase)
                
                batch_run_binoms = get_compositional_semantic_embeddings_all_layers(
                    model, tokenizer, phrase, layers=layers
                )
                
                for key, layer in layers_key.items():
                    comp_layers_names[key].append(batch_run_binoms[layer])

            # Process sentences
            timer = 0 #for debugging
            for minibatch, w1, w2 in zip(input_texts_sentences, input_texts_word1, input_texts_word2):
                timer += 1
                if timer % 100 == 0:
                        print(timer)
                if len(w1) != len(w2):
                    raise ValueError("word1 and word2 lists must be the same length")
                
                batch_run = get_semantic_representation_batch_all_layers(
                    model, tokenizer, minibatch, w1, w2, type=modeltype, layers=layers
                )
                
                for key, layer in layers_key.items():
                    layers_names[key].append(batch_run[layer])
            
            
            
            # Calculate cosine differences
            cosine_diffs = pd.DataFrame()
            
            for key in layers_key.keys():
                layer = layers_key[key]
                sentence_representations = torch.stack(layers_names[key], dim=0)
                compositional_representations = torch.stack(comp_layers_names[key], dim=0)

                sentence_representations = torch.stack(
                    [tensor.squeeze() if tensor.dim() > 1 else tensor for tensor in layers_names[key]], dim=0)
                compositional_representations = torch.stack(
                    [tensor.squeeze() if tensor.dim() > 1 else tensor for tensor in comp_layers_names[key]], dim=0)
                
                cosine_diffs_comp = []
                
                for rep1, rep2 in zip(sentence_representations, compositional_representations):
                    cosine_similarities = cosi(rep1, rep2).item()
                    cosine_diffs_comp.append(cosine_similarities)
                
                cosine_diffs_comp_df = pd.DataFrame({
                    'cosine_diffs': cosine_diffs_comp,
                    'binom': binom_order,
                    'layer': layer
                })
                
                cosine_diffs = pd.concat([cosine_diffs, cosine_diffs_comp_df], ignore_index=True)
            
            # Save results
            output_path = f"../Data/{model_name_for_saving}_{checkpoint}_compositional_cosine_diffs.csv"
            cosine_diffs.to_csv(output_path)
        
        print(f"Finished Processing {model_name} at checkpoint {checkpoint}.")
    else:
        layers_key = {f"layer{i+1}": value for i, value in enumerate(layers)}
        layers_names = {key: [] for key in layers_key.keys()}
        
        model, tokenizer = load_model(model_name)
        
        print(f"Processing {model_name} now.")
        
        # Determine model type
        if model_name in ['gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'gpt2']:
            modeltype = 'v2'
        else:
            modeltype = 'v1'
        

        comp_layers_names = {key: [] for key in layers_key.keys()}
        binom_order = []
                
        # Process compositional embeddings
        timer = 0
        for w1, w2 in zip(input_texts_word1, input_texts_word2):
            timer += 1
            if timer % 100 == 0:
                    print(timer)
            phrase = [f"{w1[i]} and {w2[i]}" for i in range(len(w1))]
            binom_order.extend(phrase)
            
            batch_run_binoms = get_compositional_semantic_embeddings_all_layers(
                model, tokenizer, phrase, layers=layers
            )
            
            for key, layer in layers_key.items():
                comp_layers_names[key].append(batch_run_binoms[layer])

        # Process sentences
        timer = 0 #for debugging
        for minibatch, w1, w2 in zip(input_texts_sentences, input_texts_word1, input_texts_word2):
            timer += 1
            if timer % 100 == 0:
                    print(timer)
            if len(w1) != len(w2):
                raise ValueError("word1 and word2 lists must be the same length")
            
            batch_run = get_semantic_representation_batch_all_layers(
                model, tokenizer, minibatch, w1, w2, type=modeltype, layers=layers
            )
            
            for key, layer in layers_key.items():
                layers_names[key].append(batch_run[layer])
        
        
        
        # Calculate cosine differences
        cosine_diffs = pd.DataFrame()
        
        for key in layers_key.keys():
            layer = layers_key[key]
            sentence_representations = torch.stack(layers_names[key], dim=0)
            compositional_representations = torch.stack(comp_layers_names[key], dim=0)

            sentence_representations = torch.stack(
                [tensor.squeeze() if tensor.dim() > 1 else tensor for tensor in layers_names[key]], dim=0)
            compositional_representations = torch.stack(
                [tensor.squeeze() if tensor.dim() > 1 else tensor for tensor in comp_layers_names[key]], dim=0)
            
            cosine_diffs_comp = []
            
            for rep1, rep2 in zip(sentence_representations, compositional_representations):
                cosine_similarities = cosi(rep1, rep2).item()
                cosine_diffs_comp.append(cosine_similarities)
            
            cosine_diffs_comp_df = pd.DataFrame({
                'cosine_diffs': cosine_diffs_comp,
                'binom': binom_order,
                'layer': layer
            })
            
            cosine_diffs = pd.concat([cosine_diffs, cosine_diffs_comp_df], ignore_index=True)
        
        # Save results
        output_path = f"../Data/{model_name_for_saving}_compositional_cosine_diffs.csv"
        cosine_diffs.to_csv(output_path)



model = 'allenai/OLMo-1B'
def get_number_of_hidden_layers(model):

    model, tokenizer = load_model(model)
    inputs = tokenizer('test sentence', padding=True, return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    with torch.no_grad(): #get number of layers in the model
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states_all = outputs.hidden_states

    num_hidden_layers = len(hidden_states_all)
    return num_hidden_layers

num_hidden_layers = get_number_of_hidden_layers(model)
layers = list(range(num_hidden_layers))

olmo1b_checkpoints = ['step96000-tokens403B', 'step95000-tokens398B'] 

main_multiple_layers(model, sentences, word1, word2, layers = layers, checkpoints=olmo1b_checkpoints)